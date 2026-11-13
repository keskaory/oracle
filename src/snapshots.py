"""the weekly-snapshot question set (d06). every rule here must be knowable on the snapshot date D."""
import pandas as pd

MIN_DAYS_LEFT = 7
MAX_DAYS_LEFT = 35
ACTIVITY_DAYS = 7    # look-back window for the liquidity check
MIN_TRADES = 5       # d22: real fills in the week before D
MIN_USD = 100.0      # d22: dollars traded in the week before D


def mondays(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    """every monday 00:00 utc between first and last, inclusive."""
    return list(pd.date_range(first.normalize(), last, freq="W-MON", tz="UTC"))


def candidate_snapshots(start: pd.Timestamp, end_scheduled: pd.Timestamp,
                        closed_time: pd.Timestamp) -> list[pd.Timestamp]:
    """mondays where the market is open and scheduled to end 7-35 days later.

    closed_time only enters as "still open at D", which you can see on D. it never decides
    *which* D we pick beyond that, so a market that resolves yes two days after D still counts.
    """
    out = []
    for d in mondays(end_scheduled - pd.Timedelta(days=MAX_DAYS_LEFT),
                     end_scheduled - pd.Timedelta(days=MIN_DAYS_LEFT)):
        if start < d < closed_time:
            out.append(d)
    return out


def price_at(history: pd.DataFrame, d: pd.Timestamp, col: str = "p") -> tuple[float, float] | None:
    """last value at or before d and its age in hours. never looks past d."""
    before = history[history["t"] <= d]
    if before.empty:
        return None
    last = before.iloc[-1]
    return float(last[col]), (d - last["t"]).total_seconds() / 3600


def activity(trades: pd.DataFrame, d: pd.Timestamp, days: int = ACTIVITY_DAYS) -> tuple[int, float]:
    """number of fills and dollars traded in the `days` before d (d itself included, nothing after)."""
    w = trades[(trades["t"] > d - pd.Timedelta(days=days)) & (trades["t"] <= d)]
    return len(w), float(w["usd"].sum())


def first_valid_snapshot(trades: pd.DataFrame, mids: pd.DataFrame,
                         candidates: list[pd.Timestamp]) -> dict | None:
    """the earliest candidate D with real trading in the week before it.

    the market price is the last real fill at or before D, not the order-book midpoint:
    an empty book has a midpoint near 0.5 that nobody actually agreed to (d22).
    """
    for d in candidates:
        n, usd = activity(trades, d)
        if n < MIN_TRADES or usd < MIN_USD:
            continue
        p, age_h = price_at(trades, d, col="yes_p")
        mid = price_at(mids, d) if len(mids) else None
        return {"snapshot": d, "market_p": p, "trade_age_h": age_h, "trades_7d": n, "usd_7d": usd,
                "mid_p": mid[0] if mid else None}
    return None
