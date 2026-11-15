"""event categories for the phase 3 subgroups, from polymarket's own event tags.

tags are messy (the event "will trump go on redscare?" is tagged dasha, anna, trump, culture, politics), so each
event gets one category by a fixed priority: the first group whose slugs appear in its tags wins.
crude on purpose: subgroups are exploratory (phase 0), and a fixed rule can't be tuned to the results.
"""
import duckdb
import pandas as pd

from cache import get_json

EVENTS = "https://gamma-api.polymarket.com/events/{}"

# most specific first: a crypto-price event is also tagged "finance", a world cup match also "world"
PRIORITY = [
    ("crypto", {"crypto", "bitcoin", "ethereum", "solana", "xrp", "crypto-prices"}),
    ("sports", {"sports", "soccer", "football", "nba", "nfl", "nhl", "mlb", "tennis", "cricket", "ufc", "f1",
                "esports", "golf", "boxing", "basketball", "baseball", "hockey", "epl", "champions-league"}),
    ("economy & stocks", {"economy", "finance", "business", "stocks", "fed", "fed-rates", "inflation",
                          "economic-policy", "earnings", "big-tech", "commodities"}),
    ("politics & world", {"politics", "elections", "world", "geopolitics", "us-politics", "global-elections",
                          "trump", "middle-east", "ukraine", "foreign-policy", "world-elections"}),
    ("culture & tech", {"pop-culture", "culture", "movies", "music", "celebrities", "tech", "ai", "science",
                        "awards", "tv", "twitter", "youtube", "mentions"}),
]
OTHER = "other"


def categorize(slugs: list[str]) -> str:
    have = {s.lower() for s in slugs}
    for name, group in PRIORITY:
        if have & group:
            return name
    return OTHER


def event_tags(event_ids: list[str]) -> pd.DataFrame:
    """one row per event: its tag slugs (joined by |) and category. every reply is cached."""
    rows = []
    for i, e in enumerate(sorted(set(event_ids)), 1):
        data = get_json(EVENTS.format(e), pause=0.2)
        slugs = [t["slug"] for t in data.get("tags") or [] if t.get("slug")]
        rows.append({"event_id": e, "tags": "|".join(slugs), "category": categorize(slugs)})
        if i % 100 == 0:
            print(f"tags: {i}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from news_sample import DB, sample_band
    con = duckdb.connect(str(DB))
    sample = sample_band(con.execute("select * from questions").df())
    t = event_tags(sample["event_id"].tolist())
    con.register("t", t)
    con.execute("create or replace table event_tags as select * from t")
    print(t["category"].value_counts().to_string())
