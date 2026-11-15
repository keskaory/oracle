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


def test_pick_numbers_match_what_the_model_sees():
    import agent
    cands = agent.pick_candidates(arts(5))
    shown = agent.pick_messages(question(), cands)[0]["content"]
    assert "1. " in shown and "5. " in shown and "6. " not in shown
    # 1-based, out-of-range and repeated numbers are handled, the order of the answer doesn't matter
    assert list(agent.picked(cands, [3, 1, 3, 99, 0]).title) == ["headline 0", "headline 2"]
    assert agent.picked(cands, []).empty


def test_news_pick_mode(monkeypatch):
    import agent
    seen = []

    def fake_chat(messages, schema):
        seen.append(schema)
        meta = {"wall_seconds": 1.0, "prompt_tokens": 10, "output_tokens": 5, "raw": "x"}
        if schema is agent.PICK_SCHEMAS[agent.PICK_VERSION]:
            return answers["pick"], meta
        return agent.Forecast(reasoning="r", probability=0.3), {**meta, "messages": messages}

    monkeypatch.setattr(agent, "chat", fake_chat)
    # the model picks headline 2 only, so the forecast prompt shows that one and not the others
    answers = {"pick": agent.PickTop(relevant=[2])}
    row = agent.forecast(question(), "news_pick", arts(5))
    assert row["parsed"] and row["agent_p"] == 0.3 and row["n_candidates"] == 5 and row["n_picked"] == 1
    # a failed pick means no forecast at all, never a silent "no news" forecast
    seen.clear()
    answers = {"pick": None}
    row = agent.forecast(question(), "news_pick", arts(5))
    assert not row["parsed"] and row["agent_p"] is None and seen == [agent.PickTop]
    # no articles at all: no pick call, the forecast says no news was found
    seen.clear()
    row = agent.forecast(question(), "news_pick", arts(0))
    assert seen == [agent.Forecast] and row["n_picked"] == 0


def test_picked_headlines_are_the_ones_shown():
    import agent
    cands = agent.pick_candidates(arts(5))
    msgs = agent.build_messages(question(), "news_pick", agent.picked(cands, [2]))
    body = text(msgs)
    assert "headline 1" in body and "headline 0" not in body and "headline 2" not in body


def test_p1_prompt_unchanged():
    # p1 results are reported, so its exact text must stay rerunnable from the cache
    import agent
    assert agent.PICK_PROMPTS["p1"].endswith("If none do, give an empty list.")
    assert agent.PickTop.model_json_schema()["properties"]["relevant"]["maxItems"] == 15


def test_news_pick_without_an_article_table(monkeypatch):
    # run_agent always passes a (maybe empty) table, but a None must not crash
    import agent
    monkeypatch.setattr(agent, "chat", lambda m, s: (agent.Forecast(reasoning="r", probability=0.2),
                                                     {"wall_seconds": 1, "prompt_tokens": 1, "output_tokens": 1,
                                                      "raw": "x"}))
    row = agent.forecast(question(), "news_pick", None)
    assert row["parsed"] and row["n_candidates"] == 0 and row["n_picked"] == 0
    assert agent.NONE_FOUND in text(agent.build_messages(question(), "news_pick", None))
