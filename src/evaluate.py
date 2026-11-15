"""phase 3: score the agent against the market, and test whether combining the two beats the market.

usage: python evaluate.py               train minus dev. fitted models are scored out of fold, so this is a
                                        preview that never touches test (d28)
       python evaluate.py --open-test   the one-time test run: fit on train minus dev, score on test (d35)

the agent is the 12b with picked headlines (d35). the other arms are reported next to it as secondary results.
every result is band-only: the news sample was drawn from questions priced 0.10-0.90 (d27), so d17's
"full set" exists for the market alone (phase 1), not for the agent.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from news_sample import DB, add_split, sample_band
from scoring import brier, ci, cluster_bootstrap, log_loss, murphy, reliability_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG = ROOT / "results", ROOT / "figures"
OPENED = OUT / "test_opened.txt"
REPS = 10_000      # phase 0: 10,000 event-bootstrap reps for the headline comparisons
SUB_REPS = 2_000   # subgroups are exploratory, fewer reps is fine
FOLDS = 5
CLIP = 0.01        # the agent sometimes says 0 or 1, and logit(0) is infinite. 0.01 caps log-odds at +-4.6

ARMS = {  # column name -> (model, mode, prompt_version)
    "agent": ("gemma3:12b", "news_pick", "v2+p2"),  # d35: the headline agent
    "agent_no_news": ("gemma3:12b", "no_news", "v2"),
    "4b_news": ("gemma3:4b", "news", "v2"),
    "4b_no_news": ("gemma3:4b", "no_news", "v2"),
}
# "will xrp dip to $1.50", "above $340", "be 4.1%", "5+ times": the answer hinges on a number the agent
# can't see today (crude regex, exploratory subgroup only)
TARGET_RE = (r"(?i)\$|\d%|\d\+|\b(?:above|below|between|reach|reaches|hit|dip|at least|more than|less than|"
             r"or more|or less|higher|lower)\b.*\d")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    s = add_split(sample_band(con.execute("select * from questions").df()))
    tags = con.execute("select event_id, category from event_tags").df()
    f = con.execute("select model, mode, prompt_version, market_id, agent_p from forecasts").df()
    con.close()
    s = s.merge(tags, on="event_id", how="left")
    for name, (model, mode, ver) in ARMS.items():
        g = f[(f["model"] == model) & (f["mode"] == mode) & (f["prompt_version"] == ver)]
        s = s.merge(g[["market_id", "agent_p"]].rename(columns={"agent_p": name}), on="market_id", how="left")
    s["week"] = s["snapshot"].dt.strftime("%Y-%m-%d")  # every snapshot is a sunday (d06), so the date is the week
    s["kind"] = np.where(s["question"].str.contains(TARGET_RE), "price or number target", "event")
    return s


def rows(s: pd.DataFrame, label: str, open_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit set, scored set). the fit set is always train minus dev, the questions the prompts weren't tuned on.
    test is only reachable with open_test, so a typo can't score it by accident."""
    fit = s[(s["split"] == "train") & ~s["dev"] & s["agent"].notna()]
    if label == "train":
        return fit, fit
    if label == "test" and open_test:
        return fit, s[(s["split"] == "test") & s["agent"].notna()]
    raise ValueError("test stays sealed unless --open-test is passed (d28)")


def features(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([logit(d[c]) for c in cols])


FITTED = {"market_recal": ["market_p"], "agent_recal": ["agent"], "combined": ["market_p", "agent"]}


def fit_predict(fit: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    """every forecaster that learns from outcomes, fit on `fit` and predicted on `apply`.
    apply's outcomes are never read."""
    y = fit["outcome"].to_numpy()
    out = pd.DataFrame(index=apply.index)
    out["base_rate"] = y.mean()
    for name, cols in FITTED.items():
        # C=1e6 is effectively no penalty, same as phase 1's calibration slope
        m = LogisticRegression(C=1e6).fit(features(fit, cols), y)
        out[name] = m.predict_proba(features(apply, cols))[:, 1]
    return out


def out_of_fold(d: pd.DataFrame) -> pd.DataFrame:
    """train preview: each question is predicted by models fit on the other folds, which never contain
    its event. scoring the fit on the rows it was fit on would flatter the combination."""
    parts = [fit_predict(d.iloc[tr], d.iloc[te])
             for tr, te in GroupKFold(n_splits=FOLDS).split(d, groups=d["event_id"])]
    return pd.concat(parts).loc[d.index]


def combination_coefs(fit: pd.DataFrame, reps: int = SUB_REPS) -> dict:
    """logit(p) = a + b*logit(market) + c*logit(agent) (d13). c = 0 means the agent adds nothing."""
    def coefs(d: pd.DataFrame) -> np.ndarray:
        m = LogisticRegression(C=1e6).fit(features(d, ["market_p", "agent"]), d["outcome"])
        return np.r_[m.intercept_, m.coef_[0]]
    a, b, c = coefs(fit)
    boot = np.array([coefs(fit.iloc[idx]) for idx in event_resamples(fit, reps, seed=2)])
    return {"a": a, "b": b, "c": c, "b_ci": ci(boot[:, 1]), "c_ci": ci(boot[:, 2])}


def event_resamples(d: pd.DataFrame, reps: int, seed: int):
    # same scheme as scoring.cluster_bootstrap, but yields row positions so a stat can return several numbers
    rng = np.random.default_rng(seed)
    groups = list(d.groupby("event_id", sort=True).indices.values())
    for _ in range(reps):
        yield np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])


def forecasters(d: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """one column per forecaster, all on the same rows."""
    p = pd.DataFrame({"market": d["market_p"], "agent": d["agent"], "constant_0.5": 0.5}, index=d.index)
    p = p.join(fitted)
    p["pool_50_50"] = sigmoid((logit(d["market_p"]) + logit(d["agent"])) / 2)  # d13: no fitting at all
    return p


def score_table(d: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """brier and log loss for every forecaster, and its brier minus the market's with event and week cis.
    the comparison that answers "does the agent add anything" is combined vs market_recal: same fitting,
    only the agent's column differs."""
    y = d["outcome"]
    loss = (p.sub(y, axis=0)) ** 2  # per-question brier, so a bootstrap is just a resampled mean
    loss["outcome"], loss["event_id"], loss["week"] = y, d["event_id"], d["week"]
    rows_ = []
    for name in p.columns:
        diff = loss[name] - loss["market"]
        r = {"forecaster": name, "n": len(d), "brier": brier(p[name], y), "log_loss": log_loss(p[name], y),
             "minus_market": float(diff.mean())}
        if name != "market":
            frame = pd.DataFrame({"x": diff, "event_id": d["event_id"], "week": d["week"]})
            r["ci_event"] = ci(cluster_bootstrap(frame, lambda f: f["x"].mean(), reps=REPS))
            r["ci_week"] = ci(cluster_bootstrap(frame, lambda f: f["x"].mean(), cluster="week", reps=REPS))
        rows_.append(r)
    # the nested test: combined minus market_recal
    frame = pd.DataFrame({"x": loss["combined"] - loss["market_recal"], "event_id": d["event_id"], "week": d["week"]})
    rows_.append({"forecaster": "combined minus market_recal", "n": len(d), "minus_market": float(frame["x"].mean()),
                  "ci_event": ci(cluster_bootstrap(frame, lambda f: f["x"].mean(), reps=REPS)),
                  "ci_week": ci(cluster_bootstrap(frame, lambda f: f["x"].mean(), cluster="week", reps=REPS))})
    return pd.DataFrame(rows_)


def secondary_arms(s_scored: pd.DataFrame) -> pd.DataFrame:
    """the other agents vs the market, each on the questions it answered (no fitting involved)."""
    out = []
    for name in ARMS:
        d = s_scored.dropna(subset=[name])
        diff = pd.DataFrame({"x": (d[name] - d["outcome"]) ** 2 - (d["market_p"] - d["outcome"]) ** 2,
                             "event_id": d["event_id"]})
        out.append({"arm": name, "n": len(d), "brier": brier(d[name], d["outcome"]),
                    "market_brier": brier(d["market_p"], d["outcome"]), "minus_market": float(diff["x"].mean()),
                    "ci_event": ci(cluster_bootstrap(diff, lambda f: f["x"].mean(), reps=REPS)),
                    "mean_p": float(d[name].mean())})
    return pd.DataFrame(out)


def subgroups(d: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """exploratory (phase 0): descriptive, with cis, not hypothesis tests."""
    g = d.assign(days=pd.qcut(d["days_left_scheduled"], 3, labels=["soonest third", "middle third", "latest third"]),
                 liquidity=pd.qcut(d["usd_7d"], 3, labels=["least traded third", "middle third", "most traded third"]))
    out = []
    for col in ["category", "kind", "days", "liquidity"]:
        for level, idx in g.groupby(col, observed=True).groups.items():
            x, q = p.loc[idx], g.loc[idx]
            diff = pd.DataFrame({"x": (x["agent"] - q["outcome"]) ** 2 - (x["market"] - q["outcome"]) ** 2,
                                 "event_id": q["event_id"]})
            out.append({"group": col, "level": level, "n": len(q), "n_events": q["event_id"].nunique(),
                        "yes_rate": q["outcome"].mean(), "market": brier(x["market"], q["outcome"]),
                        "agent": brier(x["agent"], q["outcome"]), "combined": brier(x["combined"], q["outcome"]),
                        "agent_minus_market": float(diff["x"].mean()),
                        "ci_event": ci(cluster_bootstrap(diff, lambda f: f["x"].mean(), reps=SUB_REPS))})
    return pd.DataFrame(out)


def power(d: pd.DataFrame, n_test: int) -> dict:
    """phase 0 promised the real sd of the paired difference: the smallest gap 80% power can see at n_test
    (ignores clustering, so the truth is a bit worse)."""
    sd = float((((d["agent"] - d["outcome"]) ** 2) - ((d["market_p"] - d["outcome"]) ** 2)).std())
    return {"sd_paired_diff": sd, "n": n_test, "detectable_gap": 2.8 * sd / np.sqrt(n_test)}


def plot(tables: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(4.5 * len(tables), 4.5), sharey=True)
    for ax, (name, t) in zip(axes, tables.items()):
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        ax.plot(t["mean_p"], t["freq"], marker="o")
        for r in t.itertuples():
            ax.annotate(str(r.n), (r.mean_p, r.freq), textcoords="offset points", xytext=(4, -10), fontsize=8)
        ax.set(title=name, xlabel="forecast", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("share that resolved yes")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mark_opened() -> None:
    if OPENED.exists():
        print("test was first opened:", OPENED.read_text().strip())
        return
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout
    OPENED.write_text(f"{datetime.now().isoformat(timespec='seconds')} at commit {head.strip()}\n")


def main(open_test: bool) -> None:
    label = "test" if open_test else "train"
    s = load()
    fit, d = rows(s, label, open_test)
    if open_test:
        mark_opened()
        fitted = fit_predict(fit, d)
        scored_all = s[s["split"] == "test"]
    else:
        fitted = out_of_fold(d)
        scored_all = s[(s["split"] == "train") & ~s["dev"]]
    p = forecasters(d, fitted)
    scores = score_table(d, p)
    arms = secondary_arms(scored_all)
    subs = subgroups(d, p)
    rel = {name: reliability_table(p[name], d["outcome"]) for name in ["market", "agent", "combined"]}
    summary = {
        "set": label, "note": "train = train minus dev, fitted models out of fold" if label == "train"
        else "test: fit on train minus dev, scored once on test",
        "n": len(d), "n_events": int(d["event_id"].nunique()), "yes_rate": float(d["outcome"].mean()),
        "n_without_agent_answer": int(len(scored_all) - len(d)),
        "combination_fit_on_train": combination_coefs(fit),
        "murphy": {name: murphy(p[name], d["outcome"]) for name in rel},
        "power_at_test_size": power(fit, int((s["split"] == "test").sum())),
        "scores": scores.to_dict("records"), "secondary_arms": arms.to_dict("records"),
    }
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    scores.to_csv(OUT / f"eval_{label}_scores.csv", index=False)
    arms.to_csv(OUT / f"eval_{label}_arms.csv", index=False)
    subs.to_csv(OUT / f"eval_{label}_subgroups.csv", index=False)
    pd.concat({k: v for k, v in rel.items()}, names=["forecaster"]).to_csv(OUT / f"eval_{label}_reliability.csv")
    (OUT / f"eval_{label}.json").write_text(json.dumps(summary, indent=2, default=str))
    plot(rel, FIG / f"eval_reliability_{label}.png", f"calibration on {label} (n={len(d)}, point labels = count)")
    pd.set_option("display.width", 200)
    print(scores.to_string(index=False), "\n")
    print(arms.to_string(index=False), "\n")
    print(subs.to_string(index=False), "\n")
    print(json.dumps({k: summary[k] for k in ["n", "n_events", "yes_rate", "n_without_agent_answer",
                                              "combination_fit_on_train", "power_at_test_size"]}, indent=2,
                     default=str))


if __name__ == "__main__":
    main("--open-test" in sys.argv)
