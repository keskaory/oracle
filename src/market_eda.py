"""research question 1: how calibrated is polymarket itself? also the two phase-1 data checks."""
import json
import re
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scoring import brier, ci, cluster_bootstrap, log_loss, murphy, reliability_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG = ROOT / "results", ROOT / "figures"
BAND = (0.10, 0.90)  # d17
REPS = 10_000

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_RE = re.compile(rf"\b(?:by|on|before)\s+({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?", re.I)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(ROOT / "data" / "oracle.duckdb"), read_only=True)
    return tuple(con.execute(f"select * from {t}").df() for t in ("markets", "sampled", "questions"))


def yes_rates(sampled: pd.DataFrame) -> pd.DataFrame:
    """kai's concern: does the old 'on-schedule only' design under-count yes?
    compared inside one population (the 8,000 sampled markets) so only the selection rule differs.
    'by' in the question is a rough marker for deadline questions ("will x happen by june 30?")."""
    gap_days = (sampled["closed_time"] - sampled["end_scheduled"]).dt.total_seconds().abs() / 86400
    deadline = sampled["question"].str.contains(r"\bby\b", case=False)
    rows = []
    for name, keep in [("all sampled markets (what the snapshot design draws from)", gap_days >= 0),
                       ("old design: closed within 1 day of end date", gap_days <= 1),
                       ("dropped by old design: closed early", gap_days > 1)]:
        for kind, mask in [("all questions", keep), ("deadline ('by') questions", keep & deadline)]:
            y = sampled.loc[mask, "outcome"]
            se = float(np.sqrt(y.mean() * (1 - y.mean()) / len(y)))
            rows.append({"sample": name, "questions": kind, "n": len(y), "yes_rate": y.mean(), "se": se})
    return pd.DataFrame(rows)


def end_date_edits(markets: pd.DataFrame) -> dict:
    """compare the date written in the question ("by march 31") with the api's end date.
    a big mismatch hints that endDate was moved after launch, which d06 relies on not happening often."""
    rows = []
    for m in markets.itertuples(index=False):
        hit = DATE_RE.search(m.question)
        if not hit:
            continue
        month, day, year = hit.group(1), int(hit.group(2)), hit.group(3)
        # polymarket deadlines are usually 11:59 pm eastern, which is the next day in utc
        end_et = m.end_scheduled.tz_convert("America/New_York")
        year = int(year) if year else end_et.year
        try:
            stated = pd.Timestamp(f"{month} {day} {year}", tz="America/New_York")
        except ValueError:
            continue
        rows.append(abs((end_et.normalize() - stated).days))
    gaps = pd.Series(rows)
    return {"n_with_date_in_question": int(len(gaps)),
            "share_end_date_within_1_day": float((gaps <= 1).mean()),
            "share_off_by_more_than_7_days": float((gaps > 7).mean())}


def calibration(q: pd.DataFrame, label: str) -> dict:
    p, y = q["market_p"].to_numpy(), q["outcome"].to_numpy()
    b = cluster_bootstrap(q, lambda d: brier(d["market_p"], d["outcome"]), reps=REPS)
    ll = cluster_bootstrap(q, lambda d: log_loss(d["market_p"], d["outcome"]), reps=REPS)
    m = murphy(p, y)
    return {"set": label, "n_questions": len(q), "n_events": int(q["event_id"].nunique()),
            "yes_rate": float(y.mean()), "brier": brier(p, y), "brier_ci": ci(b),
            "log_loss": log_loss(p, y), "log_loss_ci": ci(ll),
            "always_base_rate_brier": float(y.mean() * (1 - y.mean())),
            **{k: m[k] for k in ("reliability", "resolution", "uncertainty")}}


def calibration_slope(q: pd.DataFrame, reps: int = 2000) -> dict:
    """fit logit(P(yes)) = a + b * logit(price). b = 1 is perfect. b > 1 means prices are too timid,
    i.e. longshots are overpriced and favorites underpriced: the favorite-longshot bias."""
    def fit(d: pd.DataFrame) -> float:
        x = np.log(np.clip(d["market_p"], 1e-3, 1 - 1e-3) / (1 - np.clip(d["market_p"], 1e-3, 1 - 1e-3)))
        if d["outcome"].nunique() < 2:
            return np.nan
        return float(LogisticRegression(C=1e6).fit(x.to_numpy()[:, None], d["outcome"]).coef_[0, 0])
    s = cluster_bootstrap(q, fit, reps=reps, seed=1)
    s = s[~np.isnan(s)]
    return {"slope": fit(q), "slope_ci": ci(s)}


def plot(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4.5))
    for ax, (label, t) in zip(np.atleast_1d(axes), tables.items()):
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(t["mean_p"], t["freq"], "o-")
        for r in t.itertuples():
            ax.annotate(f"n={r.n}", (r.mean_p, r.freq), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="market price at snapshot", ylabel="fraction resolved yes", title=label)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    markets, sampled, q = load()
    band = q[q["market_p"].between(*BAND)]

    yr = yes_rates(sampled)
    yr.to_csv(OUT / "yes_rate_by_design.csv", index=False)
    print(yr.to_string(index=False), "\n")

    edits = end_date_edits(markets)
    print("end date vs date in question:", edits, "\n")

    cal = [calibration(q, "full"), calibration(band, "band 0.10-0.90")]
    slope = {"full": calibration_slope(q), "band": calibration_slope(band)}
    tables = {"all questions": reliability_table(q["market_p"], q["outcome"]),
              "price 0.10-0.90": reliability_table(band["market_p"], band["outcome"])}
    for name, t in tables.items():
        t.to_csv(OUT / f"market_reliability_{'full' if name == 'all questions' else 'band'}.csv", index=False)
    plot(tables, FIG / "market_calibration.png")

    # d22 sensitivity: what the order-book midpoint would have scored on the same questions
    has_mid = q.dropna(subset=["mid_p"])
    mid_check = {"n": len(has_mid), "brier_last_trade": brier(has_mid["market_p"], has_mid["outcome"]),
                 "brier_midpoint": brier(has_mid["mid_p"], has_mid["outcome"]),
                 "median_abs_gap": float((has_mid["market_p"] - has_mid["mid_p"]).abs().median())}
    print("last trade vs midpoint:", mid_check)

    summary = {"calibration": cal, "calibration_slope": slope, "end_date_check": edits, "last_trade_vs_mid": mid_check,
               "horizon_days_scheduled_median": float(q["days_left_scheduled"].median()),
               "snapshots": [str(q["snapshot"].min()), str(q["snapshot"].max())]}
    (OUT / "market_calibration.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
