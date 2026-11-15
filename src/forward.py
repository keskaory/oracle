"""phase 4: the live forward test. the same procedure as the backtest, on markets that are open right now.

usage: python forward.py forecast [--push]   pick this week's questions, fetch news, forecast, save
       python forward.py score [--push]      look up which forecasts have resolved and score them

each forecast run writes forward/predictions/YYYY-MM-DD.jsonl. with --push it commits and pushes that file
right away, so github's push time proves the forecast existed before the outcome (d19). runs that don't
happen (laptop off) are missing weeks, never backfilled.
the agent is the phase 3 primary (d35): gemma 3 12b, picked headlines, prompt v2+p2. the market price
and a combined forecast (the phase 3 model fit on train, frozen) are saved next to it.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import agent
from cache import get_json
from evaluate import logit, sigmoid
from news import GDELT_PAUSE, parse_seendate
from news_sample import BAND, DAYS_BACK, articles_for_question, event_query, fetch_once, gdelt_params
from polymarket import GAMMA, MIN_LIFETIME_VOLUME, to_utc, trades_back_to
from scoring import brier, ci, cluster_bootstrap
from snapshots import MAX_DAYS_LEFT, MIN_DAYS_LEFT, MIN_TRADES, MIN_USD, activity, price_at

ROOT = Path(__file__).resolve().parent.parent
FWD = ROOT / "forward"
PRED_DIR = FWD / "predictions"
MODEL = "gemma3:12b"
N_WEEKLY = 30        # d41: ~30 s of model time each plus the gdelt wait keeps a run to a few hours
NEWS_HOURS = 3.0     # stop retrying gdelt after this long. events still missing get no forecast that week
MARKET_URL = "https://gamma-api.polymarket.com/markets/{}"
# the backtest's prompts showed dates in duckdb's session zone (los angeles), so a monday 00:00 utc snapshot
# read "today is <sunday>". live prompts use the same zone, or the two would differ by a day in the prompt
PROMPT_TZ = "America/Los_Angeles"
_TRAIN = json.loads((ROOT / "results" / "eval_train.json").read_text())
COMBINED = _TRAIN["combination_fit_on_train"]
# the fair baseline for the combined forecast (d38): the market alone, recalibrated on the same train rows
MARKET_RECAL = _TRAIN["market_recal_fit_on_train"]
MIN_WEEKS_FOR_CI = 5  # questions from one week share a news cycle (and often an election), so also resample weeks


def snapshot_now() -> pd.Timestamp:
    """D for a live run: now, floored to the hour, in utc."""
    return pd.Timestamp.now(tz="UTC").floor("h")


def tidy_open(raw: list[dict], d: pd.Timestamp) -> pd.DataFrame:
    """binary yes/no markets that are open at D and scheduled to end 7-35 days after it (d07)."""
    rows = []
    for m in raw:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except (TypeError, ValueError):
            continue
        if outcomes != ["Yes", "No"] or len(tokens) != 2 or m.get("closed") or not m.get("acceptingOrders"):
            continue
        start, end = to_utc(m.get("startDate")), to_utc(m.get("endDate"))
        if pd.isna(start) or pd.isna(end) or not start < d:
            continue
        days_left = (end - d).total_seconds() / 86400
        if not MIN_DAYS_LEFT <= days_left <= MAX_DAYS_LEFT:
            continue
        event = (m.get("events") or [{}])[0]
        rows.append({"market_id": m["id"], "condition_id": m.get("conditionId"), "event_id": event.get("id"),
                     "event_title": event.get("title"), "question": m["question"],
                     "description": m.get("description", ""), "start": start, "end_scheduled": end,
                     "days_left_scheduled": days_left, "lifetime_volume": float(m.get("volumeNum") or 0)})
    return pd.DataFrame(rows).drop_duplicates("market_id")


def open_markets(d: pd.Timestamp) -> pd.DataFrame:
    raw, cursor = [], None
    while True:
        params = {"closed": "false", "limit": 100, "volume_num_min": MIN_LIFETIME_VOLUME,
                  "end_date_min": (d + timedelta(days=MIN_DAYS_LEFT - 1)).strftime("%Y-%m-%d"),
                  "end_date_max": (d + timedelta(days=MAX_DAYS_LEFT + 1)).strftime("%Y-%m-%d")}
        if cursor:
            params["after_cursor"] = cursor
        page = get_json(GAMMA, params, pause=0.2, asof=d.isoformat())
        raw += page["markets"]
        cursor = page.get("next_cursor")
        if not cursor or not page["markets"]:
            return tidy_open(raw, d)


def pick_questions(cands: pd.DataFrame, d: pd.Timestamp, n: int = N_WEEKLY) -> pd.DataFrame:
    """random order (seeded by the date, so a rerun picks the same ones), then the backtest's rules (d22):
    5+ real fills and $100+ traded in the week before D, price = last fill at or before D, inside the band.
    one market per event, so the week's questions aren't copies of each other (d41)."""
    out, events = [], set()
    for m in cands.sample(frac=1, random_state=int(d.strftime("%Y%m%d"))).itertuples(index=False):
        if m.event_id in events:
            continue
        trades, complete = trades_back_to(m.condition_id, d - timedelta(days=7), asof=d.isoformat())
        if not complete or trades.empty:
            continue
        trades = trades[trades["t"] <= d]  # trades after D (made while this run was going) never count
        n_fills, usd = activity(trades, d)
        got = price_at(trades, d, col="yes_p")
        if n_fills < MIN_TRADES or usd < MIN_USD or got is None or not BAND[0] <= got[0] <= BAND[1]:
            continue
        out.append({**m._asdict(), "snapshot": d, "market_p": got[0], "trade_age_h": got[1],
                    "trades_7d": n_fills, "usd_7d": usd})
        events.add(m.event_id)
        if len(out) == n:
            break
    return pd.DataFrame(out)


def fetch_news(qs: pd.DataFrame, d: pd.Timestamp, hours: float = NEWS_HOURS) -> dict:
    """market_id -> (status, articles dataframe). one gdelt query per question, window [D - 30 days, D).
    retried until done or `hours` run out, ~20 s apart as in the backtest (d21)."""
    start, end = (d - timedelta(days=DAYS_BACK)).to_pydatetime(), d.to_pydatetime()
    queries = {q.market_id: event_query(q.event_title, [q.question]) for q in qs.itertuples()}
    strict = set()
    result = {}
    deadline = time.time() + hours * 3600
    while len(result) < len(qs) and time.time() < deadline:
        for mid, query in queries.items():
            if mid in result:
                continue
            time.sleep(GDELT_PAUSE)
            status, data = fetch_once(gdelt_params(query, start, end))
            if status == "ok":
                arts = articles_for_question(data.get("articles", []), end)
                rows = [{"market_id": mid, "rank": r, "seendate": parse_seendate(a["seendate"]),
                         "title": a.get("title", ""), "domain": a.get("domain", "")} for r, a in enumerate(arts)]
                result[mid] = ("ok" if rows else "no_news",
                               pd.DataFrame(rows, columns=["market_id", "rank", "seendate", "title", "domain"]))
            elif status == "bad_query":
                if mid in strict:
                    result[mid] = ("bad_query", None)
                else:  # one retry with the backtest's stricter form, same as news_sample
                    q = qs.set_index("market_id").loc[mid]
                    queries[mid] = event_query(q["event_title"], [q["question"]], strict=True)
                    strict.add(mid)
            if time.time() >= deadline:
                break
        print(f"{datetime.now():%H:%M}  news for {len(result)} of {len(qs)}", flush=True)
    return result


def combined(market_p: float, agent_p: float) -> float:
    return float(sigmoid(COMBINED["a"] + COMBINED["b"] * logit([market_p])[0] + COMBINED["c"] * logit([agent_p])[0]))


def market_recal(market_p) -> np.ndarray:
    # computed at score time from the frozen train fit, so it needs nothing stored in the prediction files
    return sigmoid(MARKET_RECAL["a"] + MARKET_RECAL["b"] * logit(np.asarray(market_p, dtype=float)))


def record(q, news_status: str, row: dict | None) -> dict:
    out = {"market_id": q.market_id, "event_id": q.event_id, "question": q.question,
           "snapshot": q.snapshot.isoformat(), "end_scheduled": q.end_scheduled.isoformat(),
           "market_p": q.market_p, "trades_7d": q.trades_7d, "usd_7d": round(q.usd_7d, 2),
           "news_status": news_status, "model": MODEL, "prompt_version": None, "agent_p": None,
           "combined_p": None, "reasoning": None, "n_candidates": None, "n_picked": None,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if row:
        out.update({k: row.get(k) for k in ["prompt_version", "agent_p", "reasoning", "n_candidates", "n_picked"]})
        if row.get("agent_p") is not None:
            out["combined_p"] = combined(q.market_p, row["agent_p"])
    return out


def ollama_ready() -> None:
    # fail before spending hours on gdelt, not after
    names = [m["name"] for m in requests.get("http://localhost:11434/api/tags", timeout=10).json()["models"]]
    if MODEL not in names:
        raise SystemExit(f"{MODEL} is not pulled in ollama (have {names})")


def forecast(push: bool) -> None:
    ollama_ready()
    d = snapshot_now()
    path = PRED_DIR / f"{d:%Y-%m-%d}.jsonl"
    if path.exists():
        raise SystemExit(f"{path.name} already exists: one forecast run per day, never redone")
    t0 = time.time()
    cands = open_markets(d)
    qs = pick_questions(cands, d)
    print(f"D = {d}: {len(cands)} open markets in the window, {len(qs)} picked", flush=True)
    news = fetch_news(qs, d)
    agent.MODEL = MODEL
    recs = []
    shown = qs.assign(snapshot=qs["snapshot"].dt.tz_convert(PROMPT_TZ),
                      end_scheduled=qs["end_scheduled"].dt.tz_convert(PROMPT_TZ))
    for q in shown.itertuples(index=False):
        status, arts = news.get(q.market_id, ("not_fetched", None))
        if arts is not None:
            arts = arts.assign(seendate=pd.to_datetime(arts["seendate"], utc=True).dt.tz_convert(PROMPT_TZ))
        # same rule as the backtest: only questions whose news fetch finished get a forecast
        row = agent.forecast(q, "news_pick", arts) if status in ("ok", "no_news") else None
        recs.append(record(q, status, row))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    runs = FWD / "runs.csv"
    line = pd.DataFrame([{"snapshot": d.isoformat(), "open_in_window": len(cands), "picked": len(qs),
                          "forecast": sum(r["agent_p"] is not None for r in recs),
                          "minutes": round((time.time() - t0) / 60, 1)}])
    line.to_csv(runs, mode="a", header=not runs.exists(), index=False)
    print(f"wrote {path.name}: {line.iloc[0].to_dict()}", flush=True)
    if push:
        git_push([path, runs], f"forward test: forecasts for {d:%Y-%m-%d}")


def outcome_now(market_id: str, asof: str) -> tuple[str, int | None]:
    """('resolved', 0/1), ('open', None) or ('void', None) for a 50/50 refund."""
    m = get_json(MARKET_URL.format(market_id), pause=0.1, asof=asof)
    prices = json.loads(m.get("outcomePrices") or "[]")
    if not m.get("closed"):
        return "open", None
    if prices == ["1", "0"]:
        return "resolved", 1
    if prices == ["0", "1"]:
        return "resolved", 0
    return "void", None


def summarize(s: pd.DataFrame) -> dict:
    """scores on resolved questions that have an agent forecast. cis need a few events to mean anything."""
    r = s[(s["status"] == "resolved") & s["agent_p"].notna()].copy()
    out = {"resolved_with_forecast": len(r), "events": int(r["event_id"].nunique()) if len(r) else 0}
    if len(r):
        r["market_recal_p"] = market_recal(r["market_p"])
        for name in ["market_p", "market_recal_p", "agent_p", "combined_p"]:
            out[f"brier_{name[:-2]}"] = brier(r[name], r["outcome"])
        sq = {name: (r[name] - r["outcome"]) ** 2 for name in ["market_p", "market_recal_p", "agent_p", "combined_p"]}
        diffs = {"agent_minus_market": sq["agent_p"] - sq["market_p"],
                 "combined_minus_market_recal": sq["combined_p"] - sq["market_recal_p"]}
        weeks = int(r["snapshot"].nunique())
        out["weeks_resolved"] = weeks
        for key, x in diffs.items():
            r["x"] = x
            out[key] = float(x.mean())
            if out["events"] >= 10:
                out[key + "_ci"] = ci(cluster_bootstrap(r, lambda f: f["x"].mean(), reps=2000))
            if weeks >= MIN_WEEKS_FOR_CI:
                out[key + "_ci_by_week"] = ci(cluster_bootstrap(r, lambda f: f["x"].mean(), cluster="snapshot",
                                                                reps=2000))
    return out


def score(push: bool) -> None:
    files = sorted(PRED_DIR.glob("*.jsonl"))
    preds = pd.concat([pd.read_json(f, lines=True, dtype={"market_id": str, "event_id": str}) for f in files])
    asof = snapshot_now().isoformat()
    looked = [outcome_now(m, asof) for m in preds["market_id"]]
    preds["status"], preds["outcome"] = [x[0] for x in looked], [x[1] for x in looked]
    preds["scored_at"] = asof
    preds.to_csv(FWD / "scores.csv", index=False)
    summary = {"weeks": len(files), "forecasts": int(preds["agent_p"].notna().sum()),
               "status": preds["status"].value_counts().to_dict(), **summarize(preds)}
    (FWD / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    if push:
        git_push([FWD / "scores.csv", FWD / "summary.json"], f"forward test: scores as of {asof[:10]}")


def git_push(paths: list[Path], message: str) -> None:
    """stage only these files (never `git add -A`, so nothing else can ride along), commit, push."""
    rel = [str(p.relative_to(ROOT)) for p in paths]
    for cmd in (["git", "add", "--", *rel], ["git", "commit", "-m", message, "--", *rel], ["git", "push"]):
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    {"forecast": forecast, "score": score}[sys.argv[1]]("--push" in sys.argv)
