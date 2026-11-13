PY = ~/.venvs/oracle/bin/python

# everything reads from data/cache, so a rerun with a full cache makes no network calls
all: test data market

test:
	$(PY) -m pytest -q tests

data:
	cd src && $(PY) -u build.py

market:
	cd src && $(PY) -u market_eda.py

gdelt-check:
	cd src && $(PY) -u news.py

.PHONY: all test data market gdelt-check news-fetch news

# d27: one gdelt query per event for the 500-question news sample. resumable, takes several nights
news-fetch:
	cd src && caffeinate -i $(PY) -u news_sample.py fetch >> ../data/news_fetch.log 2>&1

news:
	cd src && $(PY) -u news_sample.py assemble
