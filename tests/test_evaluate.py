import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate  # noqa: E402
from categories import categorize  # noqa: E402


def fake(n_events: int = 60, per_event: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_events * per_event
    market = rng.uniform(0.1, 0.9, n)
    y = (rng.random(n) < market).astype(int)
    return pd.DataFrame({"market_id": [str(i) for i in range(n)], "event_id": np.repeat(np.arange(n_events), per_event),
                         "market_p": market, "agent": np.clip(market + rng.normal(0, 0.2, n), 0, 1),
                         "outcome": y, "week": "2026-01-04", "split": "train", "dev": False})


def test_test_set_is_sealed_without_the_flag():
    s = pd.concat([fake().assign(split="train"), fake(seed=1).assign(split="test")])
    with pytest.raises(ValueError):
        evaluate.rows(s, "test")
    fit, d = evaluate.rows(s, "test", open_test=True)
    assert (fit["split"] == "train").all() and (d["split"] == "test").all()


def test_dev_never_in_the_fit_set():
    s = fake()
    s.loc[:20, "dev"] = True
    fit, _ = evaluate.rows(s, "train")
    assert not fit["dev"].any()


def test_predictions_never_read_the_outcomes_they_predict():
    fit, apply = fake(), fake(seed=1)
    a = evaluate.fit_predict(fit, apply)
    b = evaluate.fit_predict(fit, apply.assign(outcome=1 - apply["outcome"]))
    pd.testing.assert_frame_equal(a, b)


def test_out_of_fold_ignores_a_questions_own_event():
    d = fake()
    flipped = d.copy()
    own = flipped["event_id"] == 7
    flipped.loc[own, "outcome"] = 1 - flipped.loc[own, "outcome"]
    a, b = evaluate.out_of_fold(d), evaluate.out_of_fold(flipped)
    # event 7's predictions come from folds without event 7, so flipping its outcomes can't move them
    pd.testing.assert_frame_equal(a[own], b[own])
    assert not a[~own].equals(b[~own])  # other events did see it


def test_pool_and_logit_by_hand():
    # logit(0.5) = 0, so the pool is sigmoid(logit(0.8) / 2) = sqrt(4) / (1 + sqrt(4)) = 2/3
    d = pd.DataFrame({"market_p": [0.5], "agent": [0.8]}, index=[0])
    p = evaluate.forecasters(d, pd.DataFrame(index=[0]))
    assert np.isclose(p["pool_50_50"].iloc[0], 2 / 3)
    assert np.isfinite(evaluate.logit([0.0, 1.0])).all()  # clipped


def test_combination_finds_an_informative_agent():
    # the outcome depends on a second signal only the "agent" sees, so c should be clearly positive
    rng = np.random.default_rng(3)
    n = 3000
    m, a = rng.normal(0, 1, n), rng.normal(0, 1, n)
    d = pd.DataFrame({"market_p": evaluate.sigmoid(m), "agent": evaluate.sigmoid(a),
                      "outcome": (rng.random(n) < evaluate.sigmoid(m + a)).astype(int), "event_id": np.arange(n)})
    c = evaluate.combination_coefs(d, reps=20)
    assert 0.8 < c["c"] < 1.2 and c["c_ci"][0] > 0


def test_target_questions():
    pat = evaluate.TARGET_RE
    yes = ["Will XRP dip to $1.50 in April?", "Will the November unemployment rate be 4.1%?",
           'Will Trump say "God" 5+ times?', "Will there be between 20 and 40 average daily flights?"]
    no = ["Will AfD win the second-most seats in the 2026 election?", "NAC Breda vs. AFC Ajax"]
    assert pd.Series(yes).str.contains(pat).all() and not pd.Series(no).str.contains(pat).any()


def test_category_priority():
    assert categorize(["finance", "crypto", "politics"]) == "crypto"  # most specific wins
    assert categorize(["trump", "pop-culture", "politics"]) == "politics & world"
    assert categorize(["weather"]) == "other"
    assert categorize(["Sports", "World"]) == "sports"  # case-insensitive
