import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import MAX_HEADLINES, NO_NEWS, NONE_FOUND, build_messages, format_headlines  # noqa: E402


def question(**kw):
    base = dict(market_id="1", question="Will X happen by March 31?", description="Resolves yes if X happens.",
                snapshot=pd.Timestamp("2025-03-03", tz="UTC"), end_scheduled=pd.Timestamp("2025-03-31", tz="UTC"),
                market_p=0.7314, outcome=1)
    base.update(kw)
    return SimpleNamespace(**base)


def arts(n: int) -> pd.DataFrame:
    return pd.DataFrame({"rank": range(n), "title": [f"headline {i}" for i in range(n)], "domain": "x.com",
                         "seendate": pd.date_range("2025-02-01", periods=n, freq="h", tz="UTC")})


def text(msgs) -> str:
    return " ".join(m["content"] for m in msgs)


def test_market_price_never_reaches_the_prompt():
    for mode in ["news", "no_news", "probe"]:
        t = text(build_messages(question(), mode, arts(3)))
        assert "0.73" not in t and "73%" not in t


def test_no_news_arm_differs_only_in_headlines():
    q = question()
    with_news, without = text(build_messages(q, "news", arts(3))), text(build_messages(q, "no_news"))
    assert NO_NEWS in without and "headline 0" in with_news
    assert with_news.split("Recent news")[0] == without.split("Recent news")[0]


def test_headlines_capped_deduped_newest_first():
    a = pd.concat([arts(30), arts(1)])  # a duplicate title
    lines = format_headlines(a).splitlines()
    assert len(lines) == MAX_HEADLINES
    dates = [ln.split(" | ")[0] for ln in lines]
    assert dates == sorted(dates, reverse=True)
    assert len(set(lines)) == len(lines)


def test_empty_news_is_said_out_loud():
    assert format_headlines(arts(0)) == NONE_FOUND


def test_prompt_dates_come_from_snapshot():
    t = text(build_messages(question(), "no_news"))
    assert "March 3, 2025" in t and "28 days from today" in t


def test_probe_works_without_snapshot():
    # the pre-cutoff control markets come straight from the api and have no snapshot date
    q = question()
    del q.snapshot
    assert "Will X happen" in text(build_messages(q, "probe"))


def test_reasoning_has_a_max_length_in_the_schema():
    # ollama turns the schema into a grammar; without maxLength a curly quote let v1 answers run forever
    from agent import MAX_REASONING_CHARS, Forecast
    assert Forecast.model_json_schema()["properties"]["reasoning"]["maxLength"] == MAX_REASONING_CHARS


def test_repeat_limit_is_a_parse_failure_not_a_crash(tmp_path, monkeypatch):
    import agent

    class Reply:
        status_code, text = 500, '{"error":"prediction aborted, token repeat limit reached"}'

    calls = []
    monkeypatch.setattr(agent, "cache_path", lambda url, payload: tmp_path / "c.json")
    monkeypatch.setattr(agent.requests, "post", lambda *a, **k: calls.append(1) or Reply())
    ans, meta = agent.chat([{"role": "user", "content": "x"}], agent.Forecast)
    assert ans is None and "repeat limit" in meta["raw"]
    assert len(calls) == 1  # not retried until something parses
    assert agent.chat([{"role": "user", "content": "x"}], agent.Forecast)[0] is None and len(calls) == 1  # cached
