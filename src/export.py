"""phase 5: write the small public files the dashboard and readme read, so neither needs data/ (gitignored).

usage: python export.py
writes results/forecasts.csv (one row per news-sample question, with the agent's reasoning),
results/probe.csv (the leakage probe) and results/cost.csv (tokens, laptop hours, hosted-api price).
"""
import duckdb
import pandas as pd

from agent_check import probe_summary
from evaluate import ARMS, OUT, load
from news_sample import DB
from run_agent import control_markets

# what a hosted api would have charged for the same tokens (openrouter, checked 2026-11-15), $ per million
PRICES = {"gemma3:12b": (0.05, 0.15), "gemma3:4b": (0.05, 0.10)}


def forecasts(s: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    model, mode, ver = ARMS["agent"]
    why = f[(f["model"] == model) & (f["mode"] == mode) & (f["prompt_version"] == ver)]
    d = s.merge(why[["market_id", "reasoning", "n_picked"]], on="market_id", how="left")
    d["split"] = d["split"].where(~d["dev"], "train (dev)")
    cols = ["market_id", "split", "week", "category", "kind", "days_left_scheduled", "question", "market_p",
            "agent", "agent_no_news", "outcome", "n_picked", "reasoning"]
    return d[cols].rename(columns={"week": "snapshot", "agent": "agent_p", "agent_no_news": "agent_no_news_p",
                                   "days_left_scheduled": "days_left"}).sort_values(["snapshot", "market_id"])


def probe(s: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    # accuracy needs outcomes, and the probe was reported on train only before test was opened (d34), so keep it
    train_ids = set(s.loc[s["split"] == "train", "market_id"])
    truth = s.set_index("market_id")["outcome"]
    ctrl_truth = control_markets().set_index("market_id")["outcome"]  # cached, no network
    rows = []
    for model, fm in f.groupby("model"):
        rows.append({"model": model, **probe_summary(fm[(fm["mode"] == "probe") & fm["market_id"].isin(train_ids)],
                                                     truth, "after the cutoff (train)")})
        rows.append({"model": model, **probe_summary(fm[fm["mode"] == "probe_control"], ctrl_truth,
                                                     "before the cutoff (control)")})
    return pd.DataFrame(rows).round(3)


def cost(f: pd.DataFrame) -> pd.DataFrame:
    f = f.fillna({"pick_prompt_tokens": 0, "pick_output_tokens": 0, "pick_seconds": 0})
    # news_pick makes two calls per question (pick headlines, then forecast), so both count
    f = f.assign(tokens_in=f["prompt_tokens"] + f["pick_prompt_tokens"],
                 tokens_out=f["output_tokens"] + f["pick_output_tokens"],
                 hours=(f["wall_seconds"] + f["pick_seconds"]) / 3600)
    g = f.groupby(["model", "mode", "prompt_version"], as_index=False).agg(
        questions=("market_id", "size"), parsed=("parsed", "sum"), tokens_in=("tokens_in", "sum"),
        tokens_out=("tokens_out", "sum"), laptop_hours=("hours", "sum"))
    price = g["model"].map(PRICES)
    g["hosted_usd"] = (g["tokens_in"] * price.str[0] + g["tokens_out"] * price.str[1]) / 1e6
    total = g[["questions", "parsed", "tokens_in", "tokens_out", "laptop_hours", "hosted_usd"]].sum()
    g = pd.concat([g, pd.DataFrame([{"model": "total", "mode": "", "prompt_version": "", **total}])])
    return g.astype({"questions": int, "parsed": int, "tokens_in": int, "tokens_out": int}).round(
        {"laptop_hours": 2, "hosted_usd": 3})


def main() -> None:
    s = load()
    con = duckdb.connect(str(DB), read_only=True)
    f = con.execute("select * from forecasts").df()
    con.close()
    for name, df in [("forecasts", forecasts(s, f)), ("probe", probe(s, f)), ("cost", cost(f))]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), "rows")
    print(pd.read_csv(OUT / "probe.csv").to_string(index=False))
    print(pd.read_csv(OUT / "cost.csv").to_string(index=False))


if __name__ == "__main__":
    main()
