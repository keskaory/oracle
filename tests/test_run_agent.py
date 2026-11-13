import pandas as pd
import duckdb

from run_agent import save


def test_save_mixes_forecast_and_probe_rows(tmp_path):
    # the forecasts table is created by the no_news run, then probe rows bring new columns
    db = tmp_path / "t.duckdb"
    key = {"model": "m", "prompt_version": "v1", "parsed": True}
    save(pd.DataFrame([{"market_id": "a", "mode": "no_news", "agent_p": 0.3, **key}]), db=db)
    save(pd.DataFrame([{"market_id": "a", "mode": "probe", "knows_outcome": True, "claimed_outcome": "yes", **key}]), db=db)
    # a rerun replaces rows instead of duplicating them
    save(pd.DataFrame([{"market_id": "a", "mode": "probe", "knows_outcome": False, "claimed_outcome": "unknown", **key}]), db=db)

    out = duckdb.connect(str(db)).execute("select * from forecasts order by mode").df()
    assert list(out["mode"]) == ["no_news", "probe"]
    assert out.loc[0, "agent_p"] == 0.3 and pd.isna(out.loc[0, "knows_outcome"])
    assert out.loc[1, "knows_outcome"] == False and pd.isna(out.loc[1, "agent_p"])
