"""the forecasting agent: gemma 3 4b via ollama, reading headlines from before the snapshot date.

three modes, all on the same questions:
  news     question + resolution rules + up to 15 headlines from [D - 30 days, D)
  no_news  the same prompt with the headlines left out, so the difference is what the news adds
  probe    the leakage probe: "do you already know how this resolved?" (phase 0)
the market price is never shown to the model. the agent has to be an independent second opinion,
or the combination model in phase 3 can't tell the two apart.
"""
import json
import time
from datetime import datetime

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from cache import cache_path

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"
PROMPT_VERSION = "v1"
MAX_HEADLINES = 15
MAX_RULES_CHARS = 1500  # resolution text; 75% of descriptions are shorter than this
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096}


class Forecast(BaseModel):
    reasoning: str = Field(description="two to four sentences")
    probability: float = Field(ge=0, le=1, description="probability the question resolves Yes")


class Probe(BaseModel):
    knows_outcome: bool
    outcome: str = Field(description='"Yes", "No", or "unknown"')


SYSTEM = ("You are a careful forecaster. Today is {today}. You know only what is in the question and "
          "the news below, plus general background knowledge. Do not assume you know anything that "
          "happened after today.")

TASK = """Question: {question}

Resolution rules: {rules}

The question closes on {closes} ({days} days from today).

Recent news headlines (newest first):
{headlines}

Estimate the probability that this question resolves Yes. Consider how often things like this happen, \
how much time is left, and what the news says. Give your reasoning in two to four sentences, then the \
probability as a number between 0 and 1."""

NO_NEWS = "(no news provided)"
NONE_FOUND = "(no relevant news articles were found)"

PROBE = """Question: {question}

Resolution rules: {rules}

Do you already know how this question resolved? Only say yes if you actually remember the outcome, \
not if you could guess it. If you know, give the outcome ("Yes" or "No"). If not, say "unknown"."""


def format_headlines(articles: pd.DataFrame) -> str:
    """top headlines by gdelt's relevance rank, shown newest first, duplicates (syndicated copies) removed."""
    if articles.empty:
        return NONE_FOUND
    a = articles.sort_values("rank").drop_duplicates("title").head(MAX_HEADLINES)
    a = a.sort_values("seendate", ascending=False)
    return "\n".join(f"- {r.seendate:%Y-%m-%d} | {r.domain} | {r.title.strip()}" for r in a.itertuples())


def long_date(d) -> str:
    # "March 3, 2025". the strftime flag for "day without leading zero" is mac/linux only, so build it by hand
    return f"{d:%B} {d.day}, {d.year}"


def build_messages(q, mode: str, articles: pd.DataFrame | None = None) -> list[dict]:
    rules = (q.description or "").strip()[:MAX_RULES_CHARS]
    if mode == "probe":
        # the probe has no date, so it also works on the control markets, which have no snapshot
        return [{"role": "user", "content": PROBE.format(question=q.question, rules=rules)}]
    today = q.snapshot
    headlines = NO_NEWS if mode == "no_news" else format_headlines(articles)
    days = max(0, round((q.end_scheduled - today).total_seconds() / 86400))
    return [{"role": "system", "content": SYSTEM.format(today=long_date(today))},
            {"role": "user", "content": TASK.format(question=q.question, rules=rules, headlines=headlines,
                                                     closes=long_date(q.end_scheduled), days=days)}]


def chat(messages: list[dict], schema: type[BaseModel]) -> tuple[BaseModel | None, dict]:
    """one cached call to the local model, constrained to the json schema.
    returns (parsed answer or None, metadata). a parse failure is recorded, never guessed around."""
    payload = {"model": MODEL, "messages": messages, "format": schema.model_json_schema(),
               "options": OPTIONS, "stream": False}
    path = cache_path(OLLAMA, payload)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        t0 = time.time()
        r = requests.post(OLLAMA, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        data["wall_seconds"] = time.time() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    meta = {"wall_seconds": data.get("wall_seconds"), "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"), "raw": data["message"]["content"]}
    try:
        return schema.model_validate_json(data["message"]["content"]), meta
    except ValidationError:
        return None, meta


def forecast(q, mode: str, articles: pd.DataFrame | None = None) -> dict:
    schema = Probe if mode == "probe" else Forecast
    ans, meta = chat(build_messages(q, mode, articles), schema)
    row = {"market_id": q.market_id, "mode": mode, "model": MODEL, "prompt_version": PROMPT_VERSION,
           "parsed": ans is not None, **{k: meta[k] for k in ("wall_seconds", "prompt_tokens", "output_tokens")}}
    if mode == "probe":
        row.update(knows_outcome=ans.knows_outcome if ans else None, claimed_outcome=ans.outcome if ans else None)
    else:
        row.update(agent_p=ans.probability if ans else None, reasoning=ans.reasoning if ans else meta["raw"])
    row["created"] = datetime.now().isoformat(timespec="seconds")
    return row
