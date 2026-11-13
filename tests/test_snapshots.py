import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from news import keep_before, parse_seendate  # noqa: E402
from snapshots import activity, candidate_snapshots, first_valid_snapshot, price_at  # noqa: E402

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def hist(rows):
    return pd.DataFrame({"t": [T(t) for t, _ in rows], "p": [p for _, p in rows]})


def test_price_at_never_looks_past_d():
    h = hist([("2025-03-02 10:00", 0.40), ("2025-03-03 00:00", 0.45), ("2025-03-03 01:00", 0.99)])
    p, age = price_at(h, T("2025-03-03 00:00"))
    assert p == 0.45 and age == 0  # a point exactly at d is fine, the 0.99 an hour later is not


def test_price_at_none_when_no_history_before_d():
    h = hist([("2025-03-04", 0.5)])
    assert price_at(h, T("2025-03-03")) is None


def trades(rows):
    return pd.DataFrame({"t": [T(t) for t, _, _ in rows], "yes_p": [p for _, p, _ in rows],
                         "usd": [u for _, _, u in rows]})


def test_activity_ignores_after_d():
    tr = trades([("2025-02-20", 0.5, 50), ("2025-03-01", 0.6, 30), ("2025-03-03 00:00", 0.6, 10),
                 ("2025-03-03 00:01", 0.9, 999)])
    assert activity(tr, T("2025-03-03")) == (2, 40.0)  # feb 20 is too old, 00:01 is after d


def test_early_yes_market_is_kept():
    # kai's case: scheduled to end mar 31, but it resolved yes on mar 5. feb 24 and mar 3 are both
    # still valid snapshots; only mondays after the close (mar 10+) drop out
    c = candidate_snapshots(T("2025-02-01"), T("2025-03-31"), T("2025-03-05"))
    assert c == [T("2025-02-24"), T("2025-03-03")]


def test_snapshot_window_is_7_to_35_days():
    c = candidate_snapshots(T("2025-01-01"), T("2025-03-31"), T("2025-04-01"))
    left = [(T("2025-03-31") - d).days for d in c]
    assert all(7 <= x <= 35 for x in left) and len(c) == 5  # feb 24 (35 days) through mar 24 (7 days)


def test_market_not_open_yet_is_skipped():
    assert candidate_snapshots(T("2025-03-25"), T("2025-03-31"), T("2025-04-01")) == []  # opens after the last valid monday


def test_first_valid_snapshot_uses_last_trade_not_midpoint():
    mids = hist([("2025-03-02", 0.48)])  # an empty book's midpoint
    tr = trades([(f"2025-03-0{i} 12:00", 0.10 + 0.01 * i, 40) for i in range(1, 10)])  # one $40 fill a day, mar 1-9
    got = first_valid_snapshot(tr, mids, [T("2025-03-03"), T("2025-03-10")])
    # mar 3: 2 fills, $80 -> too quiet. mar 10: 7 fills (mar 3-9), $280 -> pick, priced at the last fill (0.19)
    assert got["snapshot"] == T("2025-03-10")
    assert np.isclose(got["market_p"], 0.19) and got["mid_p"] == 0.48 and got["trades_7d"] == 7


def test_market_with_no_trades_is_never_eligible():
    mids = hist([("2025-03-02", 0.5)])
    empty = trades([])
    assert first_valid_snapshot(empty.astype({"t": "datetime64[ns, UTC]"}), mids, [T("2025-03-03")]) is None


def test_gdelt_filter_drops_articles_at_or_after_cutoff():
    arts = [{"seendate": "20250302T235959Z"}, {"seendate": "20250303T000000Z"}, {"seendate": "20250304T000000Z"}]
    kept = keep_before(arts, T("2025-03-03").to_pydatetime())
    assert [a["seendate"] for a in kept] == ["20250302T235959Z"]
    assert parse_seendate("20250303T000000Z").tzinfo is not None
