"""rebuild every table, figure and readme number from the cache, in order. no network, no model calls.

usage: python rerun.py                  (from src/) everything after the question set is built
       python rerun.py --from-scratch   also rebuild the question set (build.py). that needs the full raw
                                        polymarket cache (~9.5 gb), which stayed on the mac (d30)
ORACLE_OFFLINE=1 makes any cache miss an error, so a clean finish means the results came only from cached
api and model replies. the agent runs replay the four agents reported in the readme; the leakage probe and
the old v1 prompt runs stay in the database as they were made (rerunning them would relabel them).
"""
import os
import subprocess
import sys
import time

STEPS = [
    ["-m", "pytest", "-q", "../tests"],
    ["build.py"],                                         # markets -> weekly-snapshot questions (phase 1)
    ["market_eda.py"],                                    # market calibration
    ["news_sample.py", "assemble"],                       # the 500-question news sample and its headlines
    ["run_agent.py", "news_pick", "--model", "gemma3:12b"],  # the main agent (d35)
    ["run_agent.py", "no_news", "--model", "gemma3:12b"],
    ["run_agent.py", "news"],                             # gemma 3 4b
    ["run_agent.py", "no_news"],
    ["categories.py"],
    ["evaluate.py"],                                      # train preview
    ["evaluate.py", "--open-test"],                       # test. results/test_opened.txt keeps the first opening
    ["export.py"],
    ["readme_tables.py"],
]


def main(from_scratch: bool) -> None:
    offline = {**os.environ, "ORACLE_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    for step in STEPS if from_scratch else [s for s in STEPS if s != ["build.py"]]:
        t0 = time.time()
        print(f"\n== {' '.join(step)}", flush=True)
        # the tests fake their own network calls, so they run without the offline switch
        env = os.environ if step[0] == "-m" else offline
        subprocess.run([sys.executable, "-W", "ignore", *step], env=env, check=True)
        print(f"   {time.time() - t0:.0f} s", flush=True)
    print("\ndone. `git status` should show no changes to results/, figures/ or README.md")


if __name__ == "__main__":
    main("--from-scratch" in sys.argv)
