# works on macos and on windows (git bash or any shell with gnu make).
# without make, run the same python commands from inside src/, e.g. `python build.py`.
ifeq ($(OS),Windows_NT)
  PY ?= ../.venv/Scripts/python
  KEEP_AWAKE =
else
  PY ?= ~/.venvs/oracle/bin/python
  KEEP_AWAKE = caffeinate -i
endif

# everything reads from data/cache. rerun.py runs every step with the network switched off, so a cache
# miss fails instead of quietly making a new call
all:
	cd src && $(PY) -u rerun.py

test:
	cd src && $(PY) -m pytest -q ../tests

data:
	cd src && $(PY) -u build.py

market:
	cd src && $(PY) -u market_eda.py

gdelt-check:
	cd src && $(PY) -u news.py

# d27: one gdelt query per event for the 500-question news sample. resumable, takes about a day
news-fetch:
	cd src && $(KEEP_AWAKE) $(PY) -u news_sample.py fetch >> ../data/news_fetch.log 2>&1

news:
	cd src && $(PY) -u news_sample.py assemble

# phase 2 agent runs (ollama must be running). cached per prompt, so reruns are cheap
agent-baseline:
	cd src && $(KEEP_AWAKE) $(PY) -u run_agent.py no_news && $(PY) -u run_agent.py probe && $(PY) -u run_agent.py control

agent-check:
	cd src && $(PY) -u agent_check.py

# the private/ guard lives in hooks/ so it comes with every clone. run once per clone
hooks:
	git config core.hooksPath hooks

.PHONY: all test data market gdelt-check news-fetch news agent-baseline agent-check hooks
