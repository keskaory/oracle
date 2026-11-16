"""the dashboard. run with: streamlit run app.py
reads only small files in results/ and forward/, so it runs from a fresh clone without data/ or ollama."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RES, FWD = ROOT / "results", ROOT / "forward"
# one fixed color per forecaster on every chart (checked for color-blind separation)
COLORS = {"market": "#2a78d6", "agent": "#eb6834", "combined": "#1baf7a"}
NAMES = {"market": "market", "market_recal": "market, recalibrated", "agent": "agent (gemma 3 12b + news)",
         "agent_recal": "agent, recalibrated", "combined": "combined (market + agent)", "pool_50_50": "50/50 average",
         "base_rate": "always the yes rate", "constant_0.5": "always 50%",
         "combined minus market_recal": "combined minus recalibrated market"}

st.set_page_config(page_title="oracle", layout="wide")


@st.cache_data(ttl=3600)  # the weekly forward-test push changes forward/, so don't keep old files forever
def load() -> dict:
    d = {"scores": pd.read_csv(RES / "eval_test_scores.csv"),
         "test": json.loads((RES / "eval_test.json").read_text()),
         "rel": pd.read_csv(RES / "eval_test_reliability.csv").drop(columns="Unnamed: 1"),
         "subs": pd.read_csv(RES / "eval_test_subgroups.csv"),
         "market": json.loads((RES / "market_calibration.json").read_text()),
         "market_rel": pd.read_csv(RES / "market_reliability_full.csv"),
         "forecasts": pd.read_csv(RES / "forecasts.csv", dtype={"market_id": str}),
         "probe": pd.read_csv(RES / "probe.csv"), "cost": pd.read_csv(RES / "cost.csv")}
    files = sorted((FWD / "predictions").glob("*.jsonl"))
    d["live"] = pd.concat([pd.read_json(f, lines=True, dtype={"market_id": str}) for f in files]) if files else None
    d["live_summary"] = json.loads((FWD / "summary.json").read_text()) if (FWD / "summary.json").exists() else {}
    scores = FWD / "scores.csv"
    d["live_scores"] = pd.read_csv(scores, dtype={"market_id": str}) if scores.exists() else None
    return d


def ci_text(s) -> str:
    if not isinstance(s, str):
        return ""
    lo, hi = (float(x) for x in s.strip("()").split(","))
    return f"{lo:+.4f} to {hi:+.4f}"


def reliability_chart(df: pd.DataFrame, series: list[str]) -> alt.Chart:
    """predicted probability vs how often it happened, one point per 0.1-wide bin, sized by how many questions."""
    df = df[df["forecaster"].isin(series)].assign(label=lambda x: x["forecaster"].map(NAMES))
    scale = alt.Scale(domain=[NAMES[s] for s in series], range=[COLORS[s] for s in series])
    base = alt.Chart(df).encode(
        x=alt.X("mean_p:Q", title="forecast probability (bin average)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("freq:Q", title="share that resolved yes", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("label:N", scale=scale, title=None, legend=alt.Legend(orient="top")))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1]})).mark_line(color="#999", strokeDash=[4, 4], strokeWidth=1).encode(
        x="x:Q", y="x:Q")
    lines = base.mark_line(strokeWidth=2)
    dots = base.mark_circle(opacity=1, stroke="white", strokeWidth=2).encode(
        size=alt.Size("n:Q", title="questions", scale=alt.Scale(range=[40, 400]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="forecaster"), alt.Tooltip("mean_p:Q", title="forecast", format=".2f"),
                 alt.Tooltip("freq:Q", title="resolved yes", format=".2f"), alt.Tooltip("n:Q", title="questions")])
    return (diag + lines + dots).properties(height=380)


d = load()
scores = d["scores"].set_index("forecaster")
test = d["test"]

st.title("oracle: can a small local ai out-forecast polymarket?")
st.write(f"I asked gemma 3 (12b, running on my laptop) {test['n']} real Polymarket questions it couldn't know the "
         "answer to, gave it only news from before each forecast, and scored it against the market price at the "
         "same moment. Short answer: the market wins clearly, and adding the agent to the market doesn't help.")

c1, c2, c3 = st.columns(3)
c1.metric("market brier score (lower is better)", f"{scores.loc['market', 'brier']:.3f}")
c2.metric("agent brier score", f"{scores.loc['agent', 'brier']:.3f}",
          f"{scores.loc['agent', 'minus_market']:+.3f} vs market", delta_color="inverse")
gap = scores.loc["combined minus market_recal"]
c3.metric("combined minus recalibrated market", f"{gap['minus_market']:+.4f}")
c3.caption(f"95% ci {ci_text(gap['ci_event'])}: no detectable help from the agent")
st.caption(f"held-out test set: {test['n']} questions from {test['n_events']} events, {test['yes_rate']:.0%} "
           "resolved yes, all priced 10-90% by the market. cis are 95%, from resampling whole events 10,000 times.")

tab_res, tab_mkt, tab_fc, tab_live, tab_how = st.tabs(
    ["results", "is the market calibrated?", "browse forecasts", "live forward test", "how it works"])

with tab_res:
    st.subheader("every forecaster on the test set")
    t = d["scores"][d["scores"]["forecaster"] != "combined minus market_recal"].copy()
    t["forecaster"] = t["forecaster"].map(NAMES)
    t["95% ci of the gap"] = t["ci_event"].map(ci_text)
    st.dataframe(t[["forecaster", "brier", "log_loss", "minus_market", "95% ci of the gap"]].rename(
        columns={"log_loss": "log loss", "minus_market": "brier minus market"}).sort_values("brier").round(
        {"brier": 3, "log loss": 3, "brier minus market": 4}),
        hide_index=True, use_container_width=True)
    st.caption("recalibrated = refit with a logistic regression on the training questions. the fair test of "
               "\"does the agent add anything\" is combined vs the recalibrated market, because a fitted model can "
               "beat the raw price just by fixing its small calibration errors.")
    st.subheader("calibration on the test set")
    st.altair_chart(reliability_chart(d["rel"], ["market", "agent", "combined"]), use_container_width=True)
    st.caption("on the dashed line, a 30% forecast happens 30% of the time. the agent's forecasts bunch in the "
               "middle and it can't tell likely from unlikely questions as well as the market can.")
    st.subheader("where does the agent do better or worse? (exploratory)")
    subs = d["subs"].copy()
    subs["95% ci"] = subs["ci_event"].map(ci_text)
    st.dataframe(subs[["group", "level", "n", "market", "agent", "agent_minus_market", "95% ci"]].round(3),
                 hide_index=True, use_container_width=True)
    lo = subs["ci_event"].map(lambda s: float(s.strip("()").split(",")[1]) if isinstance(s, str) else 1.0)
    st.caption(f"agent minus market brier per group, so positive means the agent did worse. groups are small and "
               f"the cis are wide: the agent is worse in {(subs['agent_minus_market'] > 0).sum()} of {len(subs)} "
               f"groups, and clearly better (ci entirely below 0) in {(lo < 0).sum()}.")

with tab_mkt:
    full = d["market"]["calibration"][0]
    slope = d["market"]["calibration_slope"]["full"]
    st.subheader("polymarket is well calibrated")
    st.write(f"{full['n_questions']:,} resolved questions, each priced at a weekly snapshot 1 to 5 weeks before its scheduled end. a calibration slope "
             f"of 1 means perfect: it's **{slope['slope']:.2f}** (95% ci {slope['slope_ci'][0]:.2f} to "
             f"{slope['slope_ci'][1]:.2f}), so there's no sign of the favorite-longshot bias.")
    m = d["market_rel"].assign(forecaster="market")
    st.altair_chart(reliability_chart(m, ["market"]), use_container_width=True)
    st.caption(f"brier {full['brier']:.3f} vs {full['always_base_rate_brier']:.3f} for always guessing the yes rate.")

with tab_fc:
    f = d["forecasts"]
    cols = st.columns(3)
    split = cols[0].selectbox("set", ["test", "train", "train (dev)", "gap", "all"])
    cat = cols[1].selectbox("category", ["all"] + sorted(f["category"].dropna().unique()))
    text = cols[2].text_input("search questions")
    g = f if split == "all" else f[f["split"] == split]
    if cat != "all":
        g = g[g["category"] == cat]
    if text:
        g = g[g["question"].str.contains(text, case=False, regex=False)]
    st.caption(f"{len(g)} questions. click a row to read the agent's reasoning.")
    show = g[["snapshot", "question", "category", "market_p", "agent_p", "outcome"]].rename(
        columns={"market_p": "market", "agent_p": "agent", "outcome": "resolved yes"})
    pick = st.dataframe(show.round(2), hide_index=True, use_container_width=True, on_select="rerun",
                        selection_mode="single-row")
    rows = pick.selection.rows if pick else []
    if rows:
        r = g.iloc[rows[0]]
        st.markdown(f"**{r['question']}**")
        st.write(f"market {r['market_p']:.2f}, agent {r['agent_p']:.2f}, resolved "
                 f"{'yes' if r['outcome'] == 1 else 'no'}, {r['days_left']:.0f} days before the scheduled end")
        st.info(r["reasoning"] if isinstance(r["reasoning"], str) else "no answer from the agent")

with tab_live:
    st.subheader("forecasts made before the outcome existed")
    st.write("every saturday night my laptop picks 30 open markets, forecasts them the same way, and pushes the "
             "file to github right away, so github's record of the push proves the forecast came first. they get "
             "scored as "
             "they resolve.")
    s = d["live_summary"]
    if s:
        a, b, c = st.columns(3)
        a.metric("weekly runs", s.get("weeks", 0))
        b.metric("forecasts", s.get("forecasts", 0))
        c.metric("resolved so far", s.get("resolved_with_forecast", 0))
        if s.get("resolved_with_forecast"):
            st.write(f"brier on resolved questions: market {s['brier_market']:.3f}, agent {s['brier_agent']:.3f}, "
                     f"combined {s['brier_combined']:.3f}.")
    live = d["live_scores"] if d["live_scores"] is not None else d["live"]
    if live is not None:
        live = live.assign(week=pd.to_datetime(live["snapshot"], utc=True).dt.strftime("%Y-%m-%d"))
        cols = ["week", "question", "market_p", "agent_p", "combined_p"] + (
            ["status", "outcome"] if "status" in live else [])
        st.dataframe(live[cols].rename(columns={"market_p": "market", "agent_p": "agent", "combined_p": "combined"})
                     .round(2), hide_index=True, use_container_width=True)

with tab_how:
    st.subheader("how i kept the model from peeking")
    st.write("gemma 3's training data ends in august 2024, and every question here opened after september 2024. "
             "the agent only sees news published before the moment it forecasts, and it's compared with the market "
             "price at that same moment. i also asked each model outright whether it knew the outcome, on these "
             "questions and on 100 older ones it should know. the 12b claimed to know the new ones much less "
             "often than the old ones, which is what a model without leaked answers should do. the 4b claimed "
             "to know everything, old and new, so its answers to this question mean nothing.")
    st.dataframe(d["probe"].rename(columns={"says_knows": "says it knows", "accuracy_of_claims": "claims right",
                                            "n_claims": "claims"}), hide_index=True, use_container_width=True)
    st.subheader("what it cost")
    st.write("everything ran on my laptop for free. the same tokens on a hosted api would have cost about "
             f"${d['cost']['hosted_usd'].iloc[-1]:.2f}, so the real cost was "
             f"{d['cost']['laptop_hours'].iloc[-1]:.0f} hours of model time on my laptop, plus the hours spent waiting "
             "on the free news api.")
    st.dataframe(d["cost"], hide_index=True, use_container_width=True)
    st.caption("hosted prices from openrouter (gemma 3 12b $0.05 / $0.15, 4b $0.05 / $0.10 per million tokens in / "
               "out), checked 2026-11-15.")
