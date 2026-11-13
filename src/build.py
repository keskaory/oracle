"""markets -> weekly-snapshot question set -> duckdb. rerunning reads everything from data/cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

from polymarket import fetch_closed_markets, price_history, tidy_markets, trades_back_to
from snapshots import ACTIVITY_DAYS, candidate_snapshots, first_valid_snapshot

DB = Path(__file__).resolve().parent.parent / "data" / "oracle.duckdb"
SAMPLE_N = 8000   # d22: price histories for every market would take most of a day
SEED = 42


def check_market(m) -> dict | None:
    back_to = m.cands[0] - pd.Timedelta(days=ACTIVITY_DAYS)
    trades, complete = trades_back_to(m.condition_id, back_to)
    if not complete:
        # couldn't page back far enough. this is rare (0 of 100 in a test). logged by the caller
        return {"market_id": m.market_id, "incomplete": True}
    mids = price_history(m.yes_token, back_to, m.cands[-1])  # kept only to compare against (d22)
    got = first_valid_snapshot(trades, mids, m.cands)
    return None if got is None else {"market_id": m.market_id, "incomplete": False, **got}


def build_questions(markets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = markets.copy()
    markets["cands"] = [candidate_snapshots(r.start, r.end_scheduled, r.closed_time)
                        for r in markets.itertuples(index=False)]
    has_cands = markets[markets["cands"].str.len() > 0]
    # a plain random sample: it can't depend on outcomes, so it adds noise but no bias
    sample = has_cands.sample(n=min(SAMPLE_N, len(has_cands)), random_state=SEED)
    print(f"{len(markets)} binary markets, {len(has_cands)} with a candidate snapshot, sampled {len(sample)}", flush=True)

    # threads because the work is waiting on http, and each cache file is written by one thread only
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check_market, sample.itertuples(index=False)))
    found = pd.DataFrame([r for r in results if r])
    print(f"{int(found['incomplete'].sum())} markets hit the trade page cap and were dropped", flush=True)
    found = found[~found["incomplete"]].drop(columns="incomplete")
    q = found.merge(markets.drop(columns="cands"), on="market_id")
    q["days_left_scheduled"] = (q["end_scheduled"] - q["snapshot"]).dt.total_seconds() / 86400
    q["days_left_actual"] = (q["closed_time"] - q["snapshot"]).dt.total_seconds() / 86400
    sample = sample.drop(columns="cands").assign(in_questions=lambda d: d["market_id"].isin(q["market_id"]))
    return q, sample


def main() -> None:
    markets = tidy_markets(fetch_closed_markets())
    questions, sampled = build_questions(markets)
    con = duckdb.connect(str(DB))
    # register under different names: "select * from questions" would otherwise read the *old* duckdb
    # table called questions instead of this dataframe, and silently rewrite the old data (bit me once)
    for name, df in [("markets", markets), ("sampled", sampled), ("questions", questions)]:
        con.register(f"{name}_df", df)
        con.execute(f"create or replace table {name} as select * from {name}_df")
        assert con.execute(f"select count(*) from {name}").fetchone()[0] == len(df), f"{name} write mismatch"
    print(con.execute("select count(*), avg(outcome), min(snapshot), max(snapshot) from questions").fetchall())


if __name__ == "__main__":
    main()
