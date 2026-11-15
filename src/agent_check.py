"""phase 2 check: parse rates, the leakage probe, and agent vs market on TRAIN questions only.
test questions stay sealed until phase 3 (d28), so nothing here can be tuned against them."""
import duckdb
import pandas as pd

from news_sample import DB, add_split, sample_band
from scoring import brier, ci, cluster_bootstrap


def probe_summary(f: pd.DataFrame, truth: pd.Series, label: str) -> dict:
    """how often the model says it knows, and how often its claimed yes/no matches the real outcome."""
    claims = f[f["knows_outcome"] & f["claimed_outcome"].isin(["Yes", "No"])].copy()
    claims["right"] = (claims["claimed_outcome"] == "Yes").astype(int) == claims["market_id"].map(truth)
    return {"set": label, "n": len(f), "says_knows": f["knows_outcome"].mean(),
            "gives_yes_no": len(claims) / len(f), "accuracy_of_claims": claims["right"].mean() if len(claims) else None,
            "n_claims": len(claims)}


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    sample = add_split(sample_band(con.execute("select * from questions").df()))
    f = con.execute("select * from forecasts").df()
    print(f.groupby(["model", "mode", "prompt_version"]).agg(n=("parsed", "size"), parse_rate=("parsed", "mean"),
                                                    secs=("wall_seconds", "median")).to_string(), "\n")

    # leakage probe vs the pre-cutoff control
    # one row per model: the 4b and 12b answers must never be pooled.
    # accuracy needs outcomes, so the post-cutoff side uses train only: test outcomes stay sealed (d28)
    train_ids = set(sample.loc[sample["split"] == "train", "market_id"])
    truth = sample.set_index("market_id")["outcome"]
    rows = []
    for model, fm in f.groupby("model"):
        probe = fm[(fm["mode"] == "probe") & fm["market_id"].isin(train_ids)]
        if len(probe):
            rows.append({"model": model, **probe_summary(probe, truth, "sample, train (post-cutoff)")})
        ctrl = fm[fm["mode"] == "probe_control"]
        if len(ctrl):
            from run_agent import control_markets  # cached, no network
            ctrl_truth = control_markets().set_index("market_id")["outcome"]
            rows.append({"model": model, **probe_summary(ctrl, ctrl_truth, "control (resolved jan-jul 2024)")})
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    # agent vs market on train, minus the 40 dev questions the prompts were tuned on.
    # the sample is band-only, so both reporting sets are the band here
    train = sample[(sample["split"] == "train") & ~sample["dev"]]
    for (model, mode, ver), g in f[f["mode"].isin(["no_news", "news", "news_pick"])].groupby(
            ["model", "mode", "prompt_version"]):
        d = train.merge(g[["market_id", "agent_p"]], on="market_id").dropna(subset=["agent_p"])
        if d.empty:
            continue
        diff = cluster_bootstrap(d, lambda x: brier(x["agent_p"], x["outcome"]) - brier(x["market_p"], x["outcome"]),
                                 reps=2000)
        print(f"{model} {mode} {ver} train n={len(d)}  agent {brier(d['agent_p'], d['outcome']):.3f}  "
              f"market {brier(d['market_p'], d['outcome']):.3f}  diff ci {ci(diff)}  "
              f"agent p: mean {d['agent_p'].mean():.2f}, sd {d['agent_p'].std():.2f}, "
              f"share at 0.5 {(d['agent_p'] == 0.5).mean():.0%}")


if __name__ == "__main__":
    main()
