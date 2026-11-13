import hashlib
import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def cache_path(url: str, params: dict | None) -> Path:
    # key on the full request so the same call always maps to the same file
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def get_json(url: str, params: dict | None = None, pause: float = 0.0, retries: int = 4):
    """fetch json, reading from disk if we've made this exact request before."""
    path = cache_path(url, params)
    if path.exists():
        return json.loads(path.read_text())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wait = max(pause, 1.0)
    for attempt in range(retries):
        time.sleep(pause)
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            # wifi drop or the laptop waking from sleep. back off and retry instead of crashing
            time.sleep(wait)
            wait *= 2
            continue
        try:
            data = r.json()
        except ValueError:
            # gdelt answers rate limits with plain text and a 200, so non-json means "try again later".
            # never cache these, or a rerun would replay the failure forever
            data = None
        if r.status_code == 200 and data is not None:
            path.write_text(json.dumps(data))
            return data
        time.sleep(wait)
        wait *= 2
    raise RuntimeError(f"failed after {retries} tries: {url} {params}")
