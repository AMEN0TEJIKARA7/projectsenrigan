"""Champion portraits from CommunityDragon, cached on disk.

CommunityDragon mirrors the game client's assets. Two files matter:
  champion-summary.json — every champion's id and display name
  champion-icons/<id>.png — the square portrait

Portraits are fetched once and kept under <data folder>/icons, so the app only
needs the network the first time a champion is shown (a background warm-up on
launch fetches the rest). Offline, the UI simply shows no portrait.
"""

import base64
import json
import os
import re
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Optional

BASE = os.environ.get(
    "SENRIGAN_CDRAGON",
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1",
)
SUMMARY_MAX_AGE = 7 * 86400          # re-check the champion list weekly (new champions)
UA = {"User-Agent": "ProjectSenrigan/1.0 (+https://github.com/AMEN0TEJIKARA7/projectsenrigan)"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


class IconStore:
    def __init__(self, home: Path):
        self.dir = home / "icons"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict = {}          # normalised name -> champion id
        self._mem: dict = {}            # id -> data URL
        self._load_index()

    # --- champion list ------------------------------------------------------

    def _load_index(self) -> None:
        path = self.dir / "champion-summary.json"
        stale = not path.exists() or time.time() - path.stat().st_mtime > SUMMARY_MAX_AGE
        if stale:
            try:
                req = urllib.request.Request(f"{BASE}/champion-summary.json", headers=UA)
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = r.read()
                json.loads(data)                       # only keep it if it parses
                path.write_bytes(data)
            except Exception:                          # noqa: BLE001 - offline is fine
                pass
        if path.exists():
            try:
                for c in json.loads(path.read_text(encoding="utf-8")):
                    if c.get("id", -1) < 0:
                        continue
                    for key in (c.get("name"), c.get("alias")):
                        if key:
                            self._index.setdefault(_norm(key), int(c["id"]))
            except Exception:                          # noqa: BLE001
                self._index = {}

    def champion_id(self, name: str) -> Optional[int]:
        return self._index.get(_norm(name))

    # --- portraits ----------------------------------------------------------

    def icon(self, name: str) -> Optional[str]:
        """Data URL for a champion's portrait, or None when unavailable."""
        cid = self.champion_id(name)
        if cid is None:
            return None
        with self._lock:
            if cid in self._mem:
                return self._mem[cid]
        path = self.dir / f"{cid}.png"
        if not path.exists():
            try:
                req = urllib.request.Request(f"{BASE}/champion-icons/{cid}.png", headers=UA)
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = r.read()
                if not data.startswith(b"\x89PNG"):
                    return None
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            except Exception:                          # noqa: BLE001
                return None
        url = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        with self._lock:
            self._mem[cid] = url
        return url

    def warm(self, names) -> None:
        """Fetch every missing portrait in the background, politely paced."""
        def run():
            for n in names:
                cid = self.champion_id(n)
                if cid is not None and not (self.dir / f"{cid}.png").exists():
                    self.icon(n)
                    time.sleep(0.05)
        threading.Thread(target=run, daemon=True).start()
