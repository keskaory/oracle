from datetime import datetime, timedelta, timezone

from cache import get_json

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_PAUSE = 20.0  # gdelt says one request per 5 s, but 10 s spacing still drew 429s in the d18 check


def parse_seendate(s: str) -> datetime:
    # gdelt dates look like 20250114T153000Z
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def keep_before(articles: list[dict], cutoff: datetime) -> list[dict]:
    """drop anything gdelt saw at or after the cutoff. second line of defense after the query window."""
    return [a for a in articles if parse_seendate(a["seendate"]) < cutoff]


def search(query: str, before: datetime, days_back: int = 30, max_records: int = 25) -> list[dict]:
    """english articles matching query, seen in the days_back days before `before`."""
    start = before - timedelta(days=days_back)
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "hybridrel",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": before.strftime("%Y%m%d%H%M%S"),
    }
    data = get_json(GDELT_URL, params, pause=GDELT_PAUSE)
    # an empty result comes back as {} rather than {"articles": []}
    return keep_before(data.get("articles", []), before)


def gdelt_check() -> None:
    """d18: 3 topics x 3 dates. pass = at least 8 of 9 give >= 5 articles, all inside the window."""
    topics = ['"federal reserve"', "election", '"premier league"']
    dates = [datetime(2025, 2, 3, tzinfo=timezone.utc),
             datetime(2025, 11, 3, tzinfo=timezone.utc),
             datetime(2026, 9, 7, tzinfo=timezone.utc)]
    passed = 0
    for d in dates:
        for t in topics:
            try:
                arts = search(t, d)
            except RuntimeError as e:
                print(f"{d.date()}  {t:20s}  FAILED  {e}")
                continue
            seen = [parse_seendate(a["seendate"]) for a in arts]
            in_window = all(d - timedelta(days=30) <= s < d for s in seen)
            ok = len(arts) >= 5 and in_window
            passed += ok
            span = f"{min(seen).date()}..{max(seen).date()}" if seen else "-"
            print(f"{d.date()}  {t:20s}  n={len(arts):3d}  {span}  {'ok' if ok else 'FAIL'}")
            if arts:
                print(f"    e.g. {arts[0]['seendate']}  {arts[0]['title'][:80]}")
    print(f"\npassed {passed}/9 -> {'PASS' if passed >= 8 else 'FAIL'}")


if __name__ == "__main__":
    gdelt_check()
