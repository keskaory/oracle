import numpy as np
import pandas as pd

EPS = 1e-4  # clip before log loss: one market at exactly 0 that resolved yes would make the score infinite


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    """equal-width bins: mean forecast vs observed frequency, with counts so small bins are visible."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    # right edge included in the last bin so p = 1.0 isn't dropped
    df["bin"] = np.minimum((df["p"] * bins).astype(int), bins - 1)
    t = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), freq=("y", "mean")).reset_index()
    t["lo"], t["hi"] = t["bin"] / bins, (t["bin"] + 1) / bins
    return t[["lo", "hi", "n", "mean_p", "freq"]]


def murphy(p, y, bins: int = 10) -> dict:
    """brier ~= reliability - resolution + uncertainty (exact only if forecasts are constant within bins)."""
    t = reliability_table(p, y, bins)
    n, base = t["n"].sum(), float(np.mean(y))
    rel = float((t["n"] * (t["mean_p"] - t["freq"]) ** 2).sum() / n)
    res = float((t["n"] * (t["freq"] - base) ** 2).sum() / n)
    unc = base * (1 - base)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "rel_minus_res_plus_unc": rel - res + unc, "brier": brier(p, y)}


def cluster_bootstrap(df: pd.DataFrame, stat, cluster: str = "event_id",
                      reps: int = 10_000, seed: int = 0) -> np.ndarray:
    """resample whole clusters with replacement (d12) and recompute stat(df) each time."""
    rng = np.random.default_rng(seed)
    # row positions per cluster, so each rep is one fancy-index instead of concatenating thousands of frames
    groups = list(df.groupby(cluster, sort=True).indices.values())
    k = len(groups)
    out = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, k, size=k)
        out[i] = stat(df.iloc[np.concatenate([groups[j] for j in pick])])
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))
