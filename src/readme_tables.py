"""phase 5: write every number in README.md from the result files, so none is typed by hand.

usage: python readme_tables.py
the readme's prose is written by hand. numbers live only between <!-- auto:name --> and <!-- /auto --> markers,
and this script rewrites what's between them. tests/test_readme.py fails if the readme is out of date.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
README = ROOT / "README.md"
BLOCK = re.compile(r"(<!-- auto:(\w+) -->\n)(.*?)(<!-- /auto -->)", re.S)
NAMES = {"market": "market price", "combined": "combined (market + agent)",
         "market_recal": "market, recalibrated on train", "pool_50_50": "50/50 average of market and agent",
         "base_rate": "always the train yes rate", "agent_recal": "agent, recalibrated on train",
         "constant_0.5": "always 50%", "agent": "agent (gemma 3 12b, picked headlines)",
         "agent_no_news": "gemma 3 12b, no news", "4b_news": "gemma 3 4b, news", "4b_no_news": "gemma 3 4b, no news"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ci(pair, digits: int = 3) -> str:
    if isinstance(pair, str):
        pair = [float(x) for x in pair.strip("()").split(",")]
    return f"{pair[0]:+.{digits}f} to {pair[1]:+.{digits}f}"


def table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]) + "\n"


def headline() -> str:
    s = pd.read_csv(RES / "eval_test_scores.csv").set_index("forecaster")
    t = read_json(RES / "eval_test.json")
    rows = [[NAMES[f], f"{r.brier:.3f}", f"{r.log_loss:.3f}", "" if f == "market" else f"{r.minus_market:+.4f}",
             "" if f == "market" else ci(r.ci_event, 4)]
            for f, r in s.drop(index="combined minus market_recal").sort_values("brier").iterrows()]
    gap = s.loc["combined minus market_recal"]
    return (table(rows, ["forecaster", "brier", "log loss", "brier minus market", "95% ci"]) +
            f"\ntest set: {t['n']} questions from {t['n_events']} events, {t['yes_rate']:.0%} resolved yes. "
            f"{t['n_without_agent_answer']} more got no usable answer from the agent and are left out. lower is better for both scores. cis "
            f"resample whole events 10,000 times.\n\n**combined minus recalibrated market: "
            f"{gap.minus_market:+.4f} (95% ci {ci(gap.ci_event, 4)}).**\n")


def market() -> str:
    m = read_json(RES / "market_calibration.json")
    rows = []
    for c, key in zip(m["calibration"], ["full", "band"]):
        sl = m["calibration_slope"][key]
        rows.append([c["set"], f"{c['n_questions']:,}", f"{c['n_events']:,}", f"{c['yes_rate']:.0%}",
                     f"{c['brier']:.3f} ({c['brier_ci'][0]:.3f} to {c['brier_ci'][1]:.3f})",
                     f"{c['always_base_rate_brier']:.3f}",
                     f"{sl['slope']:.2f} ({sl['slope_ci'][0]:.2f} to {sl['slope_ci'][1]:.2f})"])
    return table(rows, ["questions", "n", "events", "yes rate", "market brier (95% ci)", "always the yes rate",
                        "calibration slope (95% ci)"])


def split() -> str:
    tr, te = read_json(RES / "eval_train.json"), read_json(RES / "eval_test.json")
    f = pd.read_csv(RES / "forecasts.csv")
    n = f["split"].value_counts()
    opened = (RES / "test_opened.txt").read_text().strip()
    return (f"the news sample has {len(f)} questions: {n['train'] + n['train (dev)']} in train (of which "
            f"{n['train (dev)']} were a dev set for tuning prompts), {n['test']} in test and {n['gap']} in a gap "
            f"between them that's never scored. train minus dev ({tr['n']} answered questions, {tr['n_events']} "
            f"events) is what the combination model is fit on. test ({te['n']} answered, {te['n_events']} events) "
            f"was opened once, on {opened}.\n")


def probe() -> str:
    p = pd.read_csv(RES / "probe.csv")
    rows = [[r.model, r.set, r.n, f"{r.says_knows:.0%}",
             "" if pd.isna(r.accuracy_of_claims) else f"{r.accuracy_of_claims:.0%} of {r.n_claims}"]
            for r in p.itertuples()]
    return table(rows, ["model", "questions", "n", "says it knows the outcome", "its claimed outcome was right"])


def decomposition() -> str:
    t = read_json(RES / "eval_test.json")["murphy"]
    rows = [[NAMES[k], f"{v['reliability']:.3f}", f"{v['resolution']:.3f}", f"{v['uncertainty']:.3f}",
             f"{v['brier']:.3f}"] for k, v in t.items()]
    return table(rows, ["forecaster", "reliability (lower is better)", "resolution (higher is better)",
                        "uncertainty", "brier"])


def combination() -> str:
    c = read_json(RES / "eval_test.json")["combination_fit_on_train"]
    return (f"fit on train: a = {c['a']:.2f}, b (market) = {c['b']:.2f} (95% ci {c['b_ci'][0]:.2f} to "
            f"{c['b_ci'][1]:.2f}), c (agent) = {c['c']:.2f} (95% ci {c['c_ci'][0]:.2f} to {c['c_ci'][1]:.2f}).\n")


def arms() -> str:
    a = pd.DataFrame(read_json(RES / "eval_test.json")["secondary_arms"])
    rows = [[NAMES[r.arm], r.n, f"{r.brier:.3f}", f"{r.market_brier:.3f}", f"{r.minus_market:+.3f}",
             ci(r.ci_event), f"{r.mean_p:.2f}"] for r in a.itertuples()]
    return table(rows, ["agent", "n", "brier", "market brier, same questions", "minus market", "95% ci",
                        "average forecast"])


def subgroups() -> str:
    s = pd.read_csv(RES / "eval_test_subgroups.csv")
    rows = [[r.group, r.level, r.n, f"{r.market:.3f}", f"{r.agent:.3f}", f"{r.agent_minus_market:+.3f}",
             ci(r.ci_event)] for r in s.itertuples()]
    worse = int((s["agent_minus_market"] > 0).sum())
    return (table(rows, ["split by", "group", "n", "market brier", "agent brier", "agent minus market", "95% ci"]) +
            f"\nthe agent is worse than the market in {worse} of {len(s)} groups.\n")


def power() -> str:
    p = read_json(RES / "eval_train.json")["power_at_test_size"]
    return (f"on train, the per-question difference in squared error between agent and market has a standard "
            f"deviation of {p['sd_paired_diff']:.3f}. with {p['n']} test questions, a gap smaller than about "
            f"{p['detectable_gap']:.3f} in brier would usually be missed (80% power, 5% two-sided).\n")


def cost() -> str:
    c = pd.read_csv(RES / "cost.csv").fillna("")
    rows = [[r.model, r.mode, r.prompt_version, r.questions, f"{r.tokens_in:,}", f"{r.tokens_out:,}",
             f"{r.laptop_hours:.1f}", f"${r.hosted_usd:.2f}"] for r in c.itertuples()]
    return table(rows, ["model", "run", "prompt", "questions", "tokens in", "tokens out", "laptop hours",
                        "hosted price"])


BLOCKS = {"headline": headline, "market": market, "split": split, "probe": probe, "decomposition": decomposition,
          "combination": combination, "arms": arms, "subgroups": subgroups, "power": power,
          "cost": cost}  # no forward-test block: it changes every week, so it lives on the dashboard


def render(text: str) -> str:
    missing = set(BLOCKS) - {m[1] for m in BLOCK.findall(text)}
    if missing:
        raise ValueError(f"readme is missing blocks: {sorted(missing)}")
    return BLOCK.sub(lambda m: m.group(1) + BLOCKS[m.group(2)]() + m.group(4), text)


if __name__ == "__main__":
    README.write_text(render(README.read_text(encoding="utf-8")), encoding="utf-8")
    print("readme numbers updated")
