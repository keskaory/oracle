import json
from datetime import date, datetime, timezone

import pandas as pd

from cache import get_json

GAMMA = "https://gamma-api.polymarket.com/markets/keyset"
CLOB = "https://clob.polymarket.com/prices-history"

CUTOFF = "2024-09-01"      # d02: gemma 3's training data ends aug 2024
LAST_END = "2026-11-01"    # markets must be scheduled to end before we pulled the data
MIN_LIFETIME_VOLUME = 1000  # coarse prefilter so we don't fetch price histories for ~100k dust markets (see d23)


def month_starts(first: str, last: str) -> list[str]:
    out, d = [], date.fromisoformat(first).replace(day=1)
    end = date.fromisoformat(last)
    while d < end:
        out.append(d.isoformat())
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    out.append(end.isoformat())
    return out


def fetch_closed_markets() -> list[dict]:
    """every closed market that opened after the cutoff, one calendar month of end dates at a time."""
    bounds = month_starts(CUTOFF, LAST_END)
    markets = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        cursor = None
        while True:
            params = {"closed": "true", "limit": 100, "start_date_min": CUTOFF,
                      "end_date_min": lo, "end_date_max": hi, "volume_num_min": MIN_LIFETIME_VOLUME}
            if cursor:
                params["after_cursor"] = cursor
            page = get_json(GAMMA, params, pause=0.2)
            markets += page["markets"]
            cursor = page.get("next_cursor")
            if not cursor or not page["markets"]:
                break
        print(f"{lo}..{hi}: {len(markets)} so far")
    return markets


def to_utc(s) -> pd.Timestamp:
    # gamma mixes "2025-03-14T12:00:00Z" and "2025-03-14 18:23:57+00"
    return pd.to_datetime(s, utc=True, format="mixed") if s else pd.NaT


def tidy_markets(raw: list[dict]) -> pd.DataFrame:
    """one row per binary yes/no market that resolved cleanly to 0 or 1."""
    rows = []
    for m in raw:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except (TypeError, ValueError):
            continue
        if outcomes != ["Yes", "No"] or len(tokens) != 2 or prices not in (["1", "0"], ["0", "1"]):
            # skips multi-outcome markets, unresolved ones, and 50/50 refunds
            continue
        event = (m.get("events") or [{}])[0]
        rows.append({
            "market_id": m["id"],
            "condition_id": m.get("conditionId"),
            "event_id": event.get("id"),
            "event_title": event.get("title"),
            "question": m["question"],
            "description": m.get("description", ""),
            "start": to_utc(m.get("startDate")),
            "end_scheduled": to_utc(m.get("endDate")),
            "closed_time": to_utc(m.get("closedTime")),
            "yes_token": tokens[0],
            "outcome": 1 if prices[0] == "1" else 0,
            "lifetime_volume": float(m.get("volumeNum") or 0),
            "neg_risk": bool(m.get("negRisk")),
        })
    df = pd.DataFrame(rows).drop_duplicates("market_id")
    return df


def price_history(token: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """hourly yes prices between start and end. the clob only serves windows of ~15 days for old markets, so chunk."""
    chunks, lo = [], start
    while lo < end:
        hi = min(lo + pd.Timedelta(days=14), end)
        data = get_json(CLOB, {"market": token, "startTs": int(lo.timestamp()),
                               "endTs": int(hi.timestamp()), "fidelity": 60}, pause=0.1)
        chunks += data.get("history", [])
        lo = hi
    if not chunks:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(chunks)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df.drop_duplicates("t").sort_values("t").reset_index(drop=True)


TRADES = "https://data-api.polymarket.com/v2/trades"
MAX_TRADE_PAGES = 100  # 100k trades. past this we stop and flag the market rather than page forever


def trades_back_to(condition_id: str, back_to: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    """real fills for one market, newest first, paging back until we pass `back_to`.
    the api only filters by time for single wallets, so for a market we have to walk backwards.
    returns (trades, complete). complete=False means we hit the page cap before reaching back_to."""
    rows, cursor = [], None
    for _ in range(MAX_TRADE_PAGES):
        params = {"condition": condition_id, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = get_json(TRADES, params, pause=0.1)
        rows += page.get("data") or []
        cursor = (page.get("pagination") or {}).get("next_cursor")
        if not cursor or (rows and rows[-1]["timestamp"] < back_to.timestamp()):
            break
    else:
        return _tidy_trades(rows), False
    return _tidy_trades(rows), True


def _tidy_trades(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t", "yes_p", "usd"])
    df = pd.DataFrame(rows)
    # a "no" fill at 0.30 is the same bet as a "yes" fill at 0.70
    yes_p = df["price"].where(df["outcome"] == "Yes", 1 - df["price"])
    out = pd.DataFrame({"t": pd.to_datetime(df["timestamp"], unit="s", utc=True),
                        "yes_p": yes_p, "usd": df["size"] * df["price"]})
    return out.sort_values("t").reset_index(drop=True)


if __name__ == "__main__":
    raw = fetch_closed_markets()
    df = tidy_markets(raw)
    print(len(raw), "raw ->", len(df), "binary resolved")
