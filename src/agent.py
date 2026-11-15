"""the forecasting agent: gemma 3 via ollama (4b baseline, 12b from d32), reading headlines from before
the snapshot date. the model is set with --model in run_agent.py.

modes, all on the same questions:
  news       question + resolution rules + up to 15 headlines from [D - 30 days, D)
  news_pick  the model first picks up to 15 of the top 50 headlines (d32, d33), then forecasts from those
  no_news    the same prompt with the headlines left out, so the difference is what the news adds
  probe      the leakage probe: "do you already know how this resolved?" (phase 0)
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
# v2 (the one revision, from the 40 dev questions): reasoning capped at 1,000 characters, and two lines on
# off-topic and hyped headlines. see private notes, phase 2
PROMPT_VERSION = "v2"
MAX_REASONING_CHARS = 1000  # 95% of v1 answers were under 700; stops the curly-quote runaway
MAX_HEADLINES = 15
MAX_RULES_CHARS = 1500  # resolution text; 75% of descriptions are shorter than this
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096}


class Forecast(BaseModel):
    # in v1 the model sometimes closed this string with a curly quote, which the json grammar accepts as text,
    # so it kept writing until the context was full. a max length makes the grammar end the string
    reasoning: str = Field(description="two to four sentences", max_length=MAX_REASONING_CHARS)
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
how much time is left, and what the news says. Many headlines may be about something else: ignore any \
that are not about this question. Excited or optimistic coverage is not evidence that something will \
happen by the deadline. Give your reasoning in two to four sentences, then the probability as a number \
between 0 and 1."""

NO_NEWS = "(no news provided)"
NONE_FOUND = "(no relevant news articles were found)"

# d32: in the news_pick mode the model first picks which headlines are about the question, then forecasts
PICK_CANDIDATES = 50


class Pick(BaseModel):  # p1
    relevant: list[int] = Field(description="numbers of the headlines that are about this question",
                                max_length=PICK_CANDIDATES)


class PickTop(BaseModel):  # p2
    relevant: list[int] = Field(description="numbers of the most useful headlines, at most 15",
                                max_length=MAX_HEADLINES)


PICK_HEAD = """Question: {question}

Resolution rules: {rules}

Headlines:
{headlines}

"""
# both versions are kept so every reported result can be rerun from the cache
PICK_PROMPTS = {
    # p1: on dev it kept a median of 27 of 50 (all 50 once), so the forecast's top-15 cut did the choosing, not the model
    "p1": PICK_HEAD + """Which of these headlines are about this question? A headline counts only if it is about the \
same people, team, company or event that the question is about, in the sense the question means (a football club, \
not the city it is named after; a company's stock, not its products). List the numbers of the headlines that \
count. If none do, give an empty list.""",
    # p2 (d33, the one revision): the model has to choose its 15 most useful, so its choice decides what is shown
    "p2": PICK_HEAD + """Which of these headlines would help someone predict the answer to this question? A headline \
counts only if it is about the same people, team, company or event that the question is about, in the sense the \
question means (a football club, not the city it is named after; a company's stock, not its products). Pick at \
most 15. If none would help, give an empty list.""",
}
PICK_SCHEMAS = {"p1": Pick, "p2": PickTop}
PICK_VERSION = "p2"

PROBE = """Question: {question}

Resolution rules: {rules}

Do you already know how this question resolved? Only say yes if you actually remember the outcome, \
not if you could guess it. If you know, give the outcome ("Yes" or "No"). If not, say "unknown"."""


def format_headlines(articles: pd.DataFrame) -> str:
    """top headlines by gdelt's relevance rank, shown newest first, duplicates (syndicated copies) removed."""
    if articles is None or articles.empty:
        return NONE_FOUND
    a = articles.sort_values("rank").drop_duplicates("title").head(MAX_HEADLINES)
    a = a.sort_values("seendate", ascending=False)
    return "\n".join(f"- {r.seendate:%Y-%m-%d} | {r.domain} | {r.title.strip()}" for r in a.itertuples())


def pick_candidates(articles: pd.DataFrame) -> pd.DataFrame:
    """the headlines the model chooses from: top PICK_CANDIDATES by gdelt rank, syndicated copies removed."""
    return articles.sort_values("rank").drop_duplicates("title").head(PICK_CANDIDATES).reset_index(drop=True)


def pick_messages(q, candidates: pd.DataFrame, version: str = PICK_VERSION) -> list[dict]:
    rules = (q.description or "").strip()[:MAX_RULES_CHARS]
    lines = "\n".join(f"{i}. {r.seendate:%Y-%m-%d} | {r.domain} | {r.title.strip()}"
                      for i, r in enumerate(candidates.itertuples(), 1))
    return [{"role": "user", "content": PICK_PROMPTS[version].format(question=q.question, rules=rules, headlines=lines)}]


def picked(candidates: pd.DataFrame, numbers: list[int]) -> pd.DataFrame:
    """the chosen rows. numbers are 1-based as shown; ones outside the list are ignored, repeats count once."""
    keep = sorted({n - 1 for n in numbers if 1 <= n <= len(candidates)})
    return candidates.iloc[keep]


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
        for attempt in range(3):
            r = requests.post(OLLAMA, json=payload, timeout=600)
            if r.status_code < 500 or "repeat limit" in r.text:
                break
            time.sleep(10)  # other 500s are retried so a long run doesn't die on a blip
        if r.status_code >= 500 and "repeat limit" in r.text:
            # ollama stopped the model because it was repeating itself. that's a degenerate answer, so it is
            # recorded (and cached) as a parse failure, not retried until something parses
            data = {"message": {"content": r.text}, "ollama_error": r.text}
        else:
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
    row = {"market_id": q.market_id, "mode": mode, "model": MODEL, "prompt_version": PROMPT_VERSION}
    if mode == "news_pick":
        row["prompt_version"] = f"{PROMPT_VERSION}+{PICK_VERSION}"  # forecast prompt + pick prompt
        # step 1: the model picks the relevant headlines. if that step fails, there is no forecast:
        # showing "no news" instead would quietly turn a failure into a different kind of answer
        cands = pick_candidates(articles) if articles is not None else pd.DataFrame()
        row["n_candidates"] = len(cands)
        if len(cands):
            pick, pmeta = chat(pick_messages(q, cands), PICK_SCHEMAS[PICK_VERSION])
            row.update(pick_parsed=pick is not None, pick_seconds=pmeta["wall_seconds"],
                       pick_prompt_tokens=pmeta["prompt_tokens"], pick_output_tokens=pmeta["output_tokens"])
            if pick is None:
                row.update(parsed=False, agent_p=None, reasoning="headline pick failed: " + pmeta["raw"][:500],
                           created=datetime.now().isoformat(timespec="seconds"))
                return row
            articles = picked(cands, pick.relevant)
        row["n_picked"] = len(articles) if articles is not None else 0
    schema = Probe if mode == "probe" else Forecast
    ans, meta = chat(build_messages(q, mode, articles), schema)
    row.update(parsed=ans is not None, **{k: meta[k] for k in ("wall_seconds", "prompt_tokens", "output_tokens")})
    if mode == "probe":
        row.update(knows_outcome=ans.knows_outcome if ans else None, claimed_outcome=ans.outcome if ans else None)
    else:
        row.update(agent_p=ans.probability if ans else None, reasoning=ans.reasoning if ans else meta["raw"])
    row["created"] = datetime.now().isoformat(timespec="seconds")
    return row
