"""run the agent over the news sample and store forecasts in duckdb.

usage: python run_agent.py no_news | probe | control | news | news_pick [--dev] [--model gemma3:12b]
everything is cached per prompt, so a rerun after a crash only redoes what's missing.
results are collected in memory and written at the end, so the fetcher and this never fight over the db.
"""
import sys

import duckdb
import pandas as pd

import agent
from cache import get_json
from news_sample import DB, add_split, sample_band
from polymarket import GAMMA, tidy_markets

CONTROL_N = 100


def load_sample(con) -> pd.DataFrame:
    return add_split(sample_band(con.execute("select * from questions").df()))


def control_markets() -> pd.DataFrame:
    """negative control for the leakage probe (phase 0 critic item 4): big markets that resolved
    jan-jul 2024, before gemma's training cutoff. if the probe works, the model should 'know'
    these much more often, and be right more often, than the post-cutoff sample."""
    raw, cursor = [], None
    while len(raw) < 2000:
        params = {"closed": "true", "limit": 100, "end_date_min": "2024-01-01", "end_date_max": "2024-08-01",
                  "volume_num_min": 100_000}
        if cursor:
            params["after_cursor"] = cursor
        page = get_json(GAMMA, params, pause=0.2)
        raw += page["markets"]
        cursor = page.get("next_cursor")
        if not cursor or not page["markets"]:
            break
    m = tidy_markets(raw)
    return m.sample(n=min(CONTROL_N, len(m)), random_state=0)


def run(questions: pd.DataFrame, mode: str, articles: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(questions.itertuples(index=False), 1):
        a = articles[articles["market_id"] == q.market_id] if articles is not None else None
        rows.append(agent.forecast(q, mode, a))
        if i % 25 == 0:
            print(f"{mode}: {i}/{len(questions)}", flush=True)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, table: str = "forecasts", db=DB) -> None:
    con = duckdb.connect(str(db))
    con.register("new_df", df)
    con.execute(f"create table if not exists {table} as select * from new_df where false")
    # forecast and probe rows have different columns, and the table is created by whichever runs first
    have = set(con.execute(f"select * from {table} limit 0").df().columns)
    for name, dtype in con.execute("select column_name, column_type from (describe new_df)").fetchall():
        if name not in have:
            con.execute(f'alter table {table} add column "{name}" {dtype}')
    # replace earlier rows for the same question, mode, model and prompt version
    con.execute(f"""delete from {table} t using new_df n where t.market_id = n.market_id and t.mode = n.mode
                    and t.model = n.model and t.prompt_version = n.prompt_version""")
    con.execute(f"insert into {table} by name select * from new_df")
    print(con.execute(f"select model, mode, prompt_version, count(*), avg(parsed::int) from {table} group by all order by all").df())


def main(mode: str, dev_only: bool) -> None:
    con = duckdb.connect(str(DB), read_only=True)
    sample = load_sample(con)
    articles = None
    if mode in ("news", "news_pick"):
        # only questions whose gdelt fetch is done (assembled by `make news`)
        status = con.execute("select market_id, news_status from news_sample").df()
        done = status.loc[status["news_status"].isin(["ok", "no_news"]), "market_id"]
        sample = sample[sample["market_id"].isin(done)]
        articles = con.execute("select * from news_articles").df()
    con.close()
    if dev_only:
        sample = sample[sample["dev"]]
    if mode == "control":
        out = run(control_markets(), "probe")
        out["mode"] = "probe_control"
        save(out)
        return
    print(f"{mode}: {len(sample)} questions", flush=True)
    save(run(sample, mode, articles))


if __name__ == "__main__":
    if "--model" in sys.argv:
        agent.MODEL = sys.argv[sys.argv.index("--model") + 1]  # d32: the 12b runs next to the 4b baseline
    main(sys.argv[1], "--dev" in sys.argv)
