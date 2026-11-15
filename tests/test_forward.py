import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cache  # noqa: E402
import forward  # noqa: E402

D = pd.Timestamp("2026-11-16 00:00", tz="UTC")


def test_old_cache_keys_unchanged_and_asof_changes_the_key():
    # every cached model answer and api reply from phases 1-3 must keep its file name
    url, params = "https://x.test/a", {"b": 1, "a": 2}
    old = hashlib.sha256((url + "?" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    assert cache.cache_path(url, params).name == old + ".json"
    assert cache.cache_path(url, params, asof="2026-11-16") != cache.cache_path(url, params)
    assert cache.cache_path(url, params, asof="2026-11-16") != cache.cache_path(url, params, asof="2026-11-23")


def market(**kw) -> dict:
    m = {"id": "1", "question": "q?", "outcomes": '["Yes", "No"]', "clobTokenIds": '["y", "n"]',
         "closed": False, "acceptingOrders": True, "startDate": "2026-10-01T00:00:00Z",
         "endDate": "2026-11-30T00:00:00Z", "conditionId": "c", "events": [{"id": "e1", "title": "t"}]}
    return {**m, **kw}


def test_tidy_open_applies_the_backtest_window():
    raw = [market(),
           market(id="2", endDate="2026-11-20T00:00:00Z"),       # 4 days left: too soon
           market(id="3", endDate="2027-01-10T00:00:00Z"),       # 55 days: too far
           market(id="4", startDate="2026-11-17T00:00:00Z"),     # not open yet at D
           market(id="5", outcomes='["Up", "Down"]'),             # not yes/no
           market(id="6", closed=True)]
    assert list(forward.tidy_open(raw, D)["market_id"]) == ["1"]


def trades(rows):
    return pd.DataFrame(rows, columns=["t", "yes_p", "usd"]).assign(t=lambda d: pd.to_datetime(d["t"], utc=True))


def test_pick_ignores_trades_after_d_and_takes_one_market_per_event(monkeypatch):
    week = [("2026-11-14", 0.40, 50)] * 5
    book = {"a": week + [("2026-11-16 03:00", 0.95, 500)],  # a trade after D must not set the price
            "b": week,                                        # same event as a
            "c": [("2026-11-14", 0.05, 50)] * 5,             # outside the band
            "d": [("2026-11-14", 0.50, 10)] * 5}              # only $50 traded
    monkeypatch.setattr(forward, "trades_back_to", lambda cid, back, asof="": (trades(book[cid]), True))
    cands = pd.DataFrame({"market_id": ["1", "2", "3", "4"], "condition_id": ["a", "b", "c", "d"],
                          "event_id": ["e1", "e1", "e2", "e3"]})
    got = forward.pick_questions(cands, D)
    assert len(got) == 1 and got.iloc[0]["market_p"] == 0.40 and got.iloc[0]["event_id"] == "e1"


def test_combined_uses_the_frozen_train_fit():
    c = forward.COMBINED
    expect = forward.sigmoid(c["a"] + c["b"] * forward.logit([0.3])[0] + c["c"] * forward.logit([0.7])[0])
    assert abs(forward.combined(0.3, 0.7) - expect) < 1e-12


def test_record_without_a_forecast_has_no_combined():
    q = pd.DataFrame([{"market_id": "1", "event_id": "e", "question": "q", "snapshot": D, "end_scheduled": D,
                       "market_p": 0.4, "trades_7d": 5, "usd_7d": 100.0}]).itertuples(index=False).__next__()
    r = forward.record(q, "bad_query", None)
    assert r["agent_p"] is None and r["combined_p"] is None and r["news_status"] == "bad_query"


def test_outcome_mapping(monkeypatch):
    replies = {"1": {"closed": True, "outcomePrices": '["1", "0"]'},
               "2": {"closed": True, "outcomePrices": '["0", "1"]'},
               "3": {"closed": False, "outcomePrices": '["0.6", "0.4"]'},
               "4": {"closed": True, "outcomePrices": '["0.5", "0.5"]'}}
    monkeypatch.setattr(forward, "get_json", lambda url, pause=0, asof="": replies[url.rsplit("/", 1)[1]])
    got = [forward.outcome_now(m, "x") for m in "1234"]
    assert got == [("resolved", 1), ("resolved", 0), ("open", None), ("void", None)]


def test_market_recal_uses_the_frozen_train_fit():
    m = forward.MARKET_RECAL
    expect = forward.sigmoid(m["a"] + m["b"] * forward.logit([0.3])[0])
    assert abs(forward.market_recal([0.3])[0] - expect) < 1e-12


def scored(weeks: int, per_week: int = 4) -> pd.DataFrame:
    rows = []
    for w in range(weeks):
        for i in range(per_week):
            rows.append({"status": "resolved", "event_id": f"e{w}_{i}", "snapshot": f"2026-11-{w + 1:02d}",
                         "market_p": 0.3, "agent_p": 0.6, "combined_p": 0.32, "outcome": i % 2})
    return pd.DataFrame(rows)


def test_summarize_compares_combined_to_recalibrated_market():
    s = scored(weeks=3)
    out = forward.summarize(s)
    rec = forward.market_recal([0.3])[0]
    want = ((0.32 - s["outcome"]) ** 2 - (rec - s["outcome"]) ** 2).mean()
    assert abs(out["combined_minus_market_recal"] - want) < 1e-12
    assert "agent_minus_market_ci" in out                      # 12 events
    assert "agent_minus_market_ci_by_week" not in out          # only 3 weeks


def test_summarize_adds_week_ci_after_enough_weeks_and_skips_unforecast():
    s = scored(weeks=forward.MIN_WEEKS_FOR_CI)
    s.loc[0, "agent_p"] = None                                  # no forecast that week: left out, not scored as 0
    s.loc[1, "status"] = "open"
    out = forward.summarize(s)
    assert out["resolved_with_forecast"] == len(s) - 2
    assert "combined_minus_market_recal_ci_by_week" in out
