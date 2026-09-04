"""Champion portraits, cached on disk.

Primary source is CommunityDragon (mirrors the game client's assets):
  champion-summary.json — every champion's id and display name
  champion-icons/<id>.png — the square portrait
Fallback is Riot's Data Dragon, which serves the same portraits keyed by the
champion's internal key ("MonkeyKing" for Wukong).

Portraits are fetched once and kept under <data folder>/icons, so the network
is only needed the first time a champion is shown; a background warm-up on
launch fetches the rest. Nothing here blocks the UI: the champion index loads
on a thread, and every failure is recorded in `status()` (and icons.log) so the
app can say *why* portraits are missing instead of showing blanks.
"""

import base64
import json
import os
import re
import ssl
import threading
import time
import unicodedata
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

CDRAGON = os.environ.get(
    "SENRIGAN_CDRAGON",
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1",
)
DDRAGON = os.environ.get("SENRIGAN_DDRAGON", "https://ddragon.leagueoflegends.com")
SUMMARY_MAX_AGE = 7 * 86400          # re-check the champion list weekly (new champions)
UA = {"User-Agent": "Mozilla/5.0 ProjectSenrigan/1.0 (+https://github.com/AMEN0TEJIKARA7/projectsenrigan)"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _get(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except ssl.SSLError:
        # Some Windows builds lack a usable CA bundle for Python's ssl; fall back
        # to certifi's if it is installed (PyInstaller bundles it with requests
        # or pip; harmless if absent).
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            raise
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read()


class IconStore:
    def __init__(self, home: Path):
        self.dir = home / "icons"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict = {}          # normalised name -> {"id": int, "key": str}
        self._mem: dict = {}            # id -> data URL
        self._error: str = ""
        self._source: str = ""
        self._loaded = threading.Event()
        threading.Thread(target=self._load_index, daemon=True).start()

    # --- diagnostics --------------------------------------------------------

    def _log(self, msg: str) -> None:
        try:
            with open(self.dir / "icons.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict:
        return {"ready": self._loaded.is_set(), "champions": len(self._index),
                "source": self._source, "error": self._error,
                "cached": sum(1 for p in self.dir.glob("*.png"))}

    # --- champion index -----------------------------------------------------

    def _load_index(self) -> None:
        path = self.dir / "champion-summary.json"
        stale = not path.exists() or time.time() - path.stat().st_mtime > SUMMARY_MAX_AGE
        errors = []
        if stale:
            try:
                data = _get(f"{CDRAGON}/champion-summary.json")
                entries = json.loads(data)
                norm = [{"id": int(c["id"]), "name": c.get("name", ""), "alias": c.get("alias", "")}
                        for c in entries if int(c.get("id", -1)) > 0]
                path.write_text(json.dumps({"source": "cdragon", "champions": norm}), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                errors.append(f"CommunityDragon: {e}")
                try:
                    ver = json.loads(_get(f"{DDRAGON}/api/versions.json"))[0]
                    data = json.loads(_get(f"{DDRAGON}/cdn/{ver}/data/en_US/champion.json"))["data"]
                    norm = [{"id": int(c["key"]), "name": c["name"], "alias": c["id"]} for c in data.values()]
                    path.write_text(json.dumps({"source": f"ddragon {ver}", "champions": norm}), encoding="utf-8")
                except Exception as e2:  # noqa: BLE001
                    errors.append(f"Data Dragon: {e2}")
        if path.exists():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                self._source = doc.get("source", "")
                for c in doc["champions"]:
                    for key in (c.get("name"), c.get("alias")):
                        if key:
                            self._index.setdefault(_norm(key), {"id": c["id"], "key": c.get("alias") or ""})
            except Exception as e:  # noqa: BLE001
                errors.append(f"cached champion list unreadable: {e}")
                self._index = {}
        if not self._index:
            self._error = "; ".join(errors) or "no champion list"
            self._log("index failed: " + self._error)
        else:
            self._error = ""
        self._loaded.set()

    def champion(self, name: str) -> Optional[dict]:
        self._loaded.wait(20)
        return self._index.get(_norm(name))

    # --- portraits ----------------------------------------------------------

    def icon(self, name: str) -> Optional[str]:
        """Data URL for a champion's portrait, or None when unavailable."""
        c = self.champion(name)
        if c is None:
            return None
        cid = c["id"]
        with self._lock:
            if cid in self._mem:
                return self._mem[cid]
        path = self.dir / f"{cid}.png"
        if not path.exists():
            data = None
            for url in (f"{CDRAGON}/champion-icons/{cid}.png",
                        *( [f"{DDRAGON}/cdn/{self._source.split()[1]}/img/champion/{c['key']}.png"]
                           if self._source.startswith("ddragon ") and c.get("key") else [] )):
                try:
                    d = _get(url)
                    if d.startswith(b"\x89PNG"):
                        data = d
                        break
                    self._log(f"{name}: {url} did not return a PNG")
                except Exception as e:  # noqa: BLE001
                    self._log(f"{name}: {url}: {e}")
            if data is None:
                return None
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        url = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        with self._lock:
            self._mem[cid] = url
        return url

    def warm(self, names) -> None:
        """Fetch every missing portrait in the background, politely paced."""
        def run():
            self._loaded.wait(30)
            for n in names:
                c = self.champion(n)
                if c is not None and not (self.dir / f"{c['id']}.png").exists():
                    self.icon(n)
                    time.sleep(0.05)
        threading.Thread(target=run, daemon=True).start()
