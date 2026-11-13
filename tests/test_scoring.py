import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scoring import brier, ci, cluster_bootstrap, log_loss, murphy, reliability_table  # noqa: E402


def test_brier_by_hand():
    # (0.9-1)^2 = 0.01, (0.2-0)^2 = 0.04, (0.5-1)^2 = 0.25 -> mean 0.1
    assert np.isclose(brier([0.9, 0.2, 0.5], [1, 0, 1]), 0.1)


def test_log_loss_by_hand_and_clipped():
    assert np.isclose(log_loss([0.5, 0.5], [1, 0]), np.log(2))
    assert np.isfinite(log_loss([0.0], [1]))  # clipped, not infinite


def test_reliability_bins_keep_p_equal_one():
    t = reliability_table([0.05, 0.95, 1.0], [0, 1, 1])
    assert t["n"].sum() == 3 and t.iloc[-1]["n"] == 2


def test_murphy_is_exact_when_forecasts_are_constant_per_bin():
    rng = np.random.default_rng(1)
    p = rng.choice([0.15, 0.45, 0.85], size=2000)
    y = (rng.random(2000) < p).astype(int)
    m = murphy(p, y)
    assert np.isclose(m["rel_minus_res_plus_unc"], m["brier"])


def test_perfectly_calibrated_forecasts_have_small_reliability():
    rng = np.random.default_rng(2)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert murphy(p, y)["reliability"] < 0.001


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_are_correlated():
    # 40 events x 10 identical markets each: really 40 data points, not 400
    rng = np.random.default_rng(3)
    ev = np.repeat(np.arange(40), 10)
    y = np.repeat(rng.integers(0, 2, 40), 10)
    df = pd.DataFrame({"event_id": ev, "y": y, "market_id": np.arange(400)})
    clustered = cluster_bootstrap(df, lambda d: d["y"].mean(), reps=2000)
    naive = cluster_bootstrap(df, lambda d: d["y"].mean(), cluster="market_id", reps=2000)
    assert np.std(clustered) > 2.5 * np.std(naive)  # theory says sqrt(10) ~ 3.2x


def test_bootstrap_is_reproducible():
    df = pd.DataFrame({"event_id": [1, 1, 2, 3], "y": [1, 0, 1, 0]})
    a = cluster_bootstrap(df, lambda d: d["y"].mean(), reps=200, seed=7)
    b = cluster_bootstrap(df, lambda d: d["y"].mean(), reps=200, seed=7)
    assert np.array_equal(a, b) and ci(a)[0] <= ci(a)[1]
