"""d27: the ~500-question news sample and its gdelt pull.

gdelt only lets through about 1 request in 5, so we can't query per question for all 1,167 band
questions. instead: sample 500, make one query per event (OR-ing the key names from the event and
its questions), pull that once over a window covering every snapshot in the event, and cut it
per question locally. resumable: anything fetched is cached, so rerunning picks up where it stopped.
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from cache import cache_path, refuse_if_offline
from news import GDELT_PAUSE, GDELT_URL, parse_seendate

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "oracle.duckdb"
BAD_QUERIES = ROOT / "data" / "news_bad_queries.json"
BAD_JSON_DIR = ROOT / "data" / "news_bad_json"

BAND = (0.10, 0.90)  # d17
N_SAMPLE = 500
SEED = 27            # named after the decision, so it's easy to remember
DAYS_BACK = 30       # news window before each snapshot (d10)
MAX_TERMS = 4
MAX_RECORDS = 250    # gdelt's max per query

# capitalized words that aren't names of anything
STOP = set("""will what who which when where how does do did is are was be the a an in on by before after of at to
for from with and or vs yes no top high low end monthly weekly daily price close hit reach win beat say post posts
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday q1 q2 q3 q4 inc corp ltd this that these those
winner winners victory election elections legislative vote share turnout nominee nomination democratic republican
primary exact score best chairman mainstream fc cf sc sk afc ssc acf fk cd ac""".split())
CONNECT = {"of", "de", "del", "la", "&"}  # joins 'Bank of England', 'Brighton & Hove Albion' into one name


# ---------- sampling ----------

def sample_band(questions: pd.DataFrame, n: int = N_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """random sample of band questions, stratified by snapshot quarter.
    stratifying keeps each quarter's share the same as in the full band, so the time split
    (train = early, test = late) has the same proportions in the sample as outside it."""
    band = questions[questions["market_p"].between(*BAND)].copy()
    band["stratum"] = band["snapshot"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    sizes = band["stratum"].value_counts().sort_index()
    exact = sizes / sizes.sum() * n
    take = exact.astype(int)
    # largest remainder: hand out the leftover slots to the quarters that were rounded down most
    for s in (exact - take).sort_values(ascending=False).index[: n - take.sum()]:
        take[s] += 1
    parts = [band[band["stratum"] == s].sample(n=int(k), random_state=seed) for s, k in take.items() if k > 0]
    return pd.concat(parts).sort_values("snapshot").reset_index(drop=True)


TRAIN_SHARE = 0.6  # phase 0: earlier ~60% trains, later ~40% tests
N_DEV = 40         # train questions the prompt gets revised on. everything else is never looked at while tuning


def add_split(sample: pd.DataFrame) -> pd.DataFrame:
    """d28. T = the first monday at or after the 60th percentile snapshot.
    train = resolved before T (outcome known at T), test = snapshot on or after T,
    gap = asked before T but resolved after it. gap questions are left out of the combination fit
    and the test, because their outcomes weren't known at T but their news predates it."""
    s = sample.copy()
    # duckdb hands timestamps back in local time; snapshots are mondays 00:00 utc, so work in utc
    first = s["snapshot"].dt.tz_convert("UTC").quantile(TRAIN_SHARE).normalize()
    T = next(d for d in pd.date_range(first, periods=7, freq="D") if d.dayofweek == 0)
    s["split"] = "gap"
    s.loc[s["closed_time"] < T, "split"] = "train"
    s.loc[s["snapshot"] >= T, "split"] = "test"
    dev = s[s["split"] == "train"].sample(n=N_DEV, random_state=SEED)["market_id"]
    s["dev"] = s["market_id"].isin(dev)
    s.attrs["T"] = T
    return s


# ---------- queries ----------

def key_terms(text: str) -> list[str]:
    """runs of capitalized words, e.g. 'Will Russia capture Kostyantynivka by June 30?' -> [Russia, Kostyantynivka].
    deliberately dumb: easy to explain, and the agent sees the question text anyway."""
    text = re.sub(r"\([^)]*\)", " ", text)  # drops tickers and '(HIGH)'
    words = re.findall(r"[^\s,?!:;/\"“”]+", text)
    words = [re.sub(r"'s$|’s$", "", w).strip(".-'’") for w in words] + [""]
    is_name = [bool(w) and w[0].isupper() and w.lower() not in STOP for w in words]
    terms, run = [], []
    for i, w in enumerate(words):
        if is_name[i]:
            run.append(w)
        elif run and w.lower() in CONNECT and i + 1 < len(words) and is_name[i + 1]:
            run.append(w)
        else:
            if run:
                terms.append(" ".join(run))
            run = []
    return [t for t in terms if len(t) >= 3 and not t.replace(" ", "").isdigit()]


def merge_terms(terms: list[str], max_terms: int = MAX_TERMS) -> list[str]:
    """dedupe, preferring the shorter form: 'Trump' already matches every 'Donald Trump' article."""
    kept: list[str] = []
    for t in terms:
        words = set(t.lower().split())
        if any(set(k.lower().split()) <= words for k in kept):
            continue  # an existing term already covers this one
        kept = [k for k in kept if not words <= set(k.lower().split())]
        kept.append(t)
    return kept[:max_terms]


def fallback_terms(text: str) -> list[str]:
    """no capitalized names at all (rare): use the two longest ordinary words."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", text) if w.lower() not in STOP]
    # ties broken alphabetically: set order changes between python processes, which once gave one event
    # two different queries (so assemble looked for a cache entry the fetcher never made)
    return sorted(set(words), key=lambda w: (-len(w), w))[:2]


def quote(term: str, strict: bool) -> str | None:
    """gdelt rejects quoted phrases under 5 characters ("Visa", "U.S"). in strict mode short plain
    words go in unquoted, which gdelt accepts, and short ones with punctuation are dropped."""
    if not strict or len(term) >= 5:
        return f'"{term}"'
    return term if term.isalnum() and len(term) >= 3 else None


def event_query(event_title: str, questions: list[str], strict: bool = False) -> str:
    """one gdelt query per event. event title first so its names survive the MAX_TERMS cut.
    strict=True is the retry form, used only for events whose first query gdelt rejected, so
    queries that already worked keep their cache."""
    terms = merge_terms(key_terms(event_title or "") + [t for q in questions for t in key_terms(q)])
    if not terms:
        terms = fallback_terms(f"{event_title} {' '.join(questions)}")
    quoted = [x for x in (quote(t, strict) for t in terms) if x]
    # gdelt only allows parentheses around OR'd terms, so a single term goes bare
    return quoted[0] if len(quoted) == 1 else "(" + " OR ".join(quoted) + ")"


def event_window(snapshots: pd.Series) -> tuple[datetime, datetime]:
    """from 30 days before the earliest snapshot to the latest one. never past any question's D."""
    return (snapshots.min() - timedelta(days=DAYS_BACK)).to_pydatetime(), snapshots.max().to_pydatetime()


def query_plan(sample: pd.DataFrame) -> pd.DataFrame:
    bad = load_bad()
    rows = []
    for event_id, g in sample.groupby("event_id"):
        start, end = event_window(g["snapshot"])
        args = (g["event_title"].iloc[0], list(g["question"]))
        query = event_query(*args)
        if rejected(event_id, query, bad):
            query = event_query(*args, strict=True)  # first form was rejected, use the retry form
        rows.append({"event_id": event_id, "query": query,
                     "start": start, "end": end, "n_questions": len(g)})
    return pd.DataFrame(rows)


def gdelt_params(query: str, start: datetime, end: datetime) -> dict:
    return {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
            "maxrecords": MAX_RECORDS, "sort": "hybridrel",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}


# ---------- local cut ----------

def articles_for_question(articles: list[dict], snapshot: datetime) -> list[dict]:
    """the event's articles that a forecaster on D could have read: seen in [D - 30 days, D)."""
    lo = snapshot - timedelta(days=DAYS_BACK)
    return [a for a in articles if lo <= parse_seendate(a["seendate"]) < snapshot]


# ---------- fetching ----------

def read_gdelt(status_code: int, text: str) -> tuple[str, object]:
    """sort one gdelt reply into ('ok', data), ('rate_limited', None), ('bad_json', text) or ('bad_query', message)."""
    text = text.strip()
    # rate limits come back as plain text (sometimes with a 200)
    if status_code == 429 or text.lower().startswith("please limit") or not text:
        return "rate_limited", None
    if text.startswith("{"):
        # a real result. gdelt sometimes puts raw control characters inside strings, which strict json refuses
        try:
            data = json.loads(text, strict=False)
        except ValueError:
            return "bad_json", text  # a result we can't read: retry later, never mark the query as rejected
        return ("ok", data) if status_code == 200 else ("rate_limited", None)
    return "bad_query", text[:200]  # any other text is gdelt rejecting the query


def fetch_once(params: dict) -> tuple[str, object]:
    """one try. returns ('ok', data), ('rate_limited', None), ('bad_json', text) or ('bad_query', message).
    separate from cache.get_json because here a rejected query must not be retried forever."""
    path = cache_path(GDELT_URL, params)
    if path.exists():
        return "ok", json.loads(path.read_text())
    refuse_if_offline(GDELT_URL)
    try:
        r = requests.get(GDELT_URL, params=params, timeout=60)
    except requests.RequestException:
        return "rate_limited", None  # network blip: treat like a rate limit and come back later
    status, data = read_gdelt(r.status_code, r.text)
    if status == "ok":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return status, data


def is_cached(params: dict) -> bool:
    return cache_path(GDELT_URL, params).exists()


def load_bad() -> dict:
    """event id -> {"rejected": [queries gdelt refused], "error": last message}."""
    bad = json.loads(BAD_QUERIES.read_text()) if BAD_QUERIES.exists() else {}
    for v in bad.values():
        if "query" in v:  # older format had a single query
            v["rejected"] = [v.pop("query")]
    return bad


def rejected(event_id, query: str, bad: dict) -> bool:
    return query in bad.get(str(event_id), {}).get("rejected", [])


def fetch_all(sample: pd.DataFrame) -> None:
    """loop over unfetched events until all are done. ~20 s between requests, as in d21."""
    bad = load_bad()
    tries = ok = 0
    streak = 0
    while True:
        # rebuilt every 25 tries (not once per pass): rejected events get their retry form, and refused
        # dev events go back to the front instead of waiting hours for the next full pass
        plan = query_plan(sample)
        pending = [r for r in plan.itertuples(index=False)
                   if not rejected(r.event_id, r.query, bad) and not is_cached(gdelt_params(r.query, r.start, r.end))]
        # dev events first so prompt work can start, then train, then test
        dev_events = set(sample.loc[sample["dev"], "event_id"])
        train_events = set(sample.loc[sample["split"] == "train", "event_id"])
        pending.sort(key=lambda r: (r.event_id not in dev_events, r.event_id not in train_events))
        print(f"{datetime.now():%m-%d %H:%M}  {len(pending)} events left", flush=True)
        if not pending:
            return
        for r in pending:
            time.sleep(GDELT_PAUSE)
            status, data = fetch_once(gdelt_params(r.query, r.start, r.end))
            tries += 1
            if status == "ok":
                ok += 1
                streak = 0
            elif status == "bad_query":
                entry = bad.setdefault(str(r.event_id), {"rejected": []})
                entry["rejected"].append(r.query)
                entry["error"] = data
                BAD_QUERIES.write_text(json.dumps(bad, indent=1))
                print(f"  bad query {r.query}: {data}", flush=True)
            elif status == "bad_json":
                # keep the reply so it can be looked at; the event stays pending and is tried again
                BAD_JSON_DIR.mkdir(parents=True, exist_ok=True)
                (BAD_JSON_DIR / f"{r.event_id}.txt").write_text(data, encoding="utf-8")
                print(f"  unreadable json for {r.query}, saved, will retry", flush=True)
            else:
                streak += 1
                if streak >= 30:
                    # a long run of refusals usually means a temporary block. waiting it out beats hammering
                    print(f"{datetime.now():%m-%d %H:%M}  30 refusals in a row, pausing 15 min", flush=True)
                    time.sleep(900)
                    streak = 0
            if tries % 25 == 0:
                print(f"{datetime.now():%m-%d %H:%M}  tries {tries}  ok {ok}  ({ok / tries:.0%})", flush=True)
                break


# ---------- assembling ----------

def assemble(sample: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """per-question news status + one row per (question, article). safe to rerun mid-fetch."""
    bad = load_bad()
    by_event = plan.set_index("event_id")
    status_rows, art_rows = [], []
    for q in sample.itertuples(index=False):
        p = by_event.loc[q.event_id]
        params = gdelt_params(p["query"], p["start"], p["end"])
        if rejected(q.event_id, p["query"], bad):
            status, arts = "bad_query", []
        elif not is_cached(params):
            status, arts = "not_fetched", []
        else:
            arts = articles_for_question(fetch_once(params)[1].get("articles", []), q.snapshot.to_pydatetime())
            # no news is kept in the sample and flagged, not dropped (d27)
            status = "ok" if arts else "no_news"
        status_rows.append({"market_id": q.market_id, "news_query": p["query"], "news_status": status,
                            "n_articles": len(arts)})
        for rank, a in enumerate(arts):
            art_rows.append({"market_id": q.market_id, "rank": rank, "seendate": parse_seendate(a["seendate"]),
                             "title": a.get("title", ""), "domain": a.get("domain", ""), "url": a.get("url", "")})
    news = sample.merge(pd.DataFrame(status_rows), on="market_id")
    return news, pd.DataFrame(art_rows, columns=["market_id", "rank", "seendate", "title", "domain", "url"])


def main(cmd: str) -> None:
    con = duckdb.connect(str(DB))
    sample = add_split(sample_band(con.execute("select * from questions").df()))
    plan = query_plan(sample)
    if cmd == "plan":
        print(f"{len(sample)} questions, {len(plan)} events, T = {sample.attrs['T']}")
        print(sample.groupby(["split", "dev"]).size().to_string())
        print(sample.groupby("stratum").size().to_string())
        print(plan[["n_questions", "query"]].head(40).to_string())
    elif cmd == "fetch":
        con.close()  # don't hold the db open for days
        fetch_all(sample)
    elif cmd == "assemble":
        news, arts = assemble(sample, plan)
        for name, df in [("news_sample", news), ("news_articles", arts)]:
            con.register(f"{name}_df", df)
            con.execute(f"create or replace table {name} as select * from {name}_df")
            assert con.execute(f"select count(*) from {name}").fetchall()[0][0] == len(df), f"{name} write mismatch"
        print(news["news_status"].value_counts().to_string())
        got = news[news["news_status"].isin(["ok", "no_news"])]
        if len(got):
            print(f"articles per fetched question: median {got['n_articles'].median():.0f}, "
                  f"zero for {(got['n_articles'] == 0).mean():.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
