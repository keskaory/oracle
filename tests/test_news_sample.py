import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import datetime, timezone

import pandas as pd

from news_sample import articles_for_question, event_query, event_window, key_terms, merge_terms, read_gdelt, sample_band


def fake_questions(n_per_quarter: dict[str, int]) -> pd.DataFrame:
    rows, i = [], 0
    for quarter, n in n_per_quarter.items():
        start = pd.Period(quarter).start_time.tz_localize("UTC")
        for k in range(n):
            rows.append({"market_id": str(i), "snapshot": start + pd.Timedelta(days=k % 80),
                         "market_p": 0.5 if k % 3 else 0.95})  # every third question is outside the band
            i += 1
    return pd.DataFrame(rows)


def test_sample_keeps_quarter_shares():
    q = fake_questions({"2025Q1": 300, "2025Q2": 600, "2025Q3": 900})
    s = sample_band(q, n=100, seed=1)
    assert len(s) == 100
    assert s["market_p"].between(0.10, 0.90).all()
    shares = s["stratum"].value_counts(normalize=True)
    assert abs(shares["2025Q1"] - 1 / 6) < 0.02 and abs(shares["2025Q3"] - 0.5) < 0.02


def test_sample_is_reproducible():
    q = fake_questions({"2025Q1": 200, "2025Q2": 200})
    assert list(sample_band(q, n=50, seed=3)["market_id"]) == list(sample_band(q, n=50, seed=3)["market_id"])


def test_key_terms_skip_generic_words_and_dates():
    assert key_terms("Will Russia capture Kostyantynivka by June 30?") == ["Russia", "Kostyantynivka"]
    assert key_terms("Will Brighton & Hove Albion FC win on 2025-03-08?") == ["Brighton & Hove Albion"]


def test_merge_prefers_shorter_name():
    assert merge_terms(["Donald Trump", "Trump", "Truth Social"]) == ["Trump", "Truth Social"]


def test_single_term_has_no_parentheses():
    # gdelt rejects parentheses that don't wrap an OR
    assert event_query("Powerball jackpot", []) == '"Powerball"'
    assert event_query("Will Russia capture Kostyantynivka?", []) == '("Russia" OR "Kostyantynivka")'


def test_window_never_passes_last_snapshot():
    snaps = pd.Series(pd.to_datetime(["2025-03-03", "2025-03-17"], utc=True))
    start, end = event_window(snaps)
    assert end == datetime(2025, 3, 17, tzinfo=timezone.utc)
    assert start == datetime(2025, 2, 1, tzinfo=timezone.utc)


def test_local_cut_is_per_question():
    # one event fetch covers two snapshots. the earlier question must not see articles from between them
    arts = [{"seendate": d} for d in ["20250130T000000Z", "20250210T000000Z", "20250302T235900Z",
                                      "20250303T000000Z", "20250310T000000Z"]]
    got = articles_for_question(arts, datetime(2025, 3, 3, tzinfo=timezone.utc))
    assert [a["seendate"] for a in got] == ["20250210T000000Z", "20250302T235900Z"]


def test_strict_form_unquotes_short_words():
    assert event_query("Visa earnings", [], strict=True) == "Visa"
    # 'U.S' can't go in unquoted, so it's dropped rather than sent
    assert event_query("Will the U.S and European Union agree?", [], strict=True) == '"European Union"'
    assert event_query("Will the U.S and European Union agree?", []) == '("U.S" OR "European Union")'


def test_split_has_no_overlap():
    from news_sample import add_split
    snaps = pd.date_range("2025-01-06", periods=100, freq="W-MON", tz="UTC")
    s = pd.DataFrame({"market_id": [str(i) for i in range(100)], "snapshot": snaps,
                      "closed_time": snaps + pd.Timedelta(days=20)})
    s = add_split(s)
    T = s.attrs["T"]
    assert T.dayofweek == 0 and str(T.tz) == "UTC"
    assert (s.loc[s["split"] == "train", "closed_time"] < T).all()
    assert (s.loc[s["split"] == "test", "snapshot"] >= T).all()
    assert s.loc[s["dev"], "split"].eq("train").all()


def test_read_gdelt_sorts_replies():
    ok = '{"articles": [{"title": "a"}]}'
    assert read_gdelt(200, ok) == ("ok", {"articles": [{"title": "a"}]})
    # a raw tab inside a title broke strict parsing, and the result was wrongly marked as a rejected query
    assert read_gdelt(200, '{"articles": [{"title": "a\tb"}]}')[0] == "ok"
    # unreadable but clearly a result: retry, never reject the query
    assert read_gdelt(200, '{"articles": [ {"title": "a"')[0] == "bad_json"
    assert read_gdelt(429, "anything") == ("rate_limited", None)
    assert read_gdelt(200, "Please limit requests to one every 5 seconds") == ("rate_limited", None)
    assert read_gdelt(200, "") == ("rate_limited", None)
    assert read_gdelt(200, "The specified phrase is too short.") == ("bad_query", "The specified phrase is too short.")
