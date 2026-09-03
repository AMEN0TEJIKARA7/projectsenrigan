"""Retraining pipeline for the desktop app.

Runs the repository's own scripts — `ingest` → `train_final` → `check_skew` —
inside the app process against a per-user data folder (SENRIGAN_HOME), and
swaps the model artifact in only when the skew check passes. The scripts are
imported, not re-implemented, so the app trains exactly what the CLI trains.

Data comes from one of two places:
  * the GitHub repository's committed CSVs (default; each year is a plain
    file under 100 MB, so raw.githubusercontent.com serves it directly), or
  * CSV files the user picks from disk (a fresh Oracle's Elixir download).

Everything here runs on a worker thread; `Update.status()` is what the UI polls.
"""

import contextlib
import io
import os
import re
import shutil
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ARTIFACT_NAME = "lol_logistic_sigmoid_v1.joblib"
RAW_RE = re.compile(r"^(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv$")
GITHUB_RAW = "https://raw.githubusercontent.com/AMEN0TEJIKARA7/projectsenrigan/main/{name}"
GITHUB_YEARS = [2022, 2023, 2024, 2025, 2026]


def _src_dir() -> Path:
    """The repository's `src/` — beside `app/` in a checkout, bundled in a build."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "src"


class _LineWriter(io.TextIOBase):
    """Turns the scripts' print() output into log lines for the UI."""

    def __init__(self, sink):
        self.sink, self.buf = sink, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.sink(line.rstrip())
        return len(s)

    def flush(self):
        pass


@dataclass
class Update:
    """One retraining run; `status()` is safe to call from any thread."""

    home: Path
    log: list = field(default_factory=list)
    stage: str = "idle"
    running: bool = False
    error: str = ""
    done: bool = False
    started: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> dict:
        with self._lock:
            return {"running": self.running, "stage": self.stage, "error": self.error,
                    "done": self.done, "log": list(self.log[-60:]), "started": self.started}

    def _say(self, line: str) -> None:
        with self._lock:
            self.log.append(line)

    def _stage(self, name: str) -> None:
        with self._lock:
            self.stage = name
        self._say(f"── {name}")

    # --- data sources -------------------------------------------------------

    def fetch_github(self) -> None:
        """Download the yearly CSVs the repository has committed, skipping any
        already present at the same size (raw.githubusercontent.com sends
        Content-Length, which is enough to tell an updated file from a stale one)."""
        self._stage("downloading data")
        for year in GITHUB_YEARS:
            name = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
            dest = self.home / name
            url = GITHUB_RAW.format(name=name)
            req = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    size = int(r.headers.get("Content-Length") or 0)
            except Exception as e:  # noqa: BLE001 - surfaced to the user as text
                if dest.exists():
                    self._say(f"{name}: offline, keeping local copy ({e})")
                    continue
                raise RuntimeError(f"could not reach GitHub for {name}: {e}") from e
            if dest.exists() and size and dest.stat().st_size == size:
                self._say(f"{name}: unchanged")
                continue
            self._say(f"{name}: downloading {size / 1e6:.0f} MB")
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(tmp, dest)

    def import_files(self, paths: list) -> None:
        """Copy user-chosen Oracle's Elixir CSVs into the data folder."""
        self._stage("importing files")
        for p in map(Path, paths):
            m = RAW_RE.match(p.name)
            if not m:
                year = _sniff_year(p)
                if year is None:
                    raise RuntimeError(f"{p.name}: not an Oracle's Elixir match-data CSV")
                name = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
            else:
                name = p.name
            dest = self.home / name
            if p.resolve() != dest.resolve():
                shutil.copyfile(p, dest)
            self._say(f"{name}: imported ({dest.stat().st_size / 1e6:.0f} MB)")

    # --- the run ------------------------------------------------------------

    def run(self, source: str = "github", files: list = None) -> None:
        with self._lock:
            if self.running:
                return
            self.running, self.done, self.error = True, False, ""
            self.log.clear()
            self.started = datetime.now().strftime("%H:%M")
        threading.Thread(target=self._run, args=(source, files or []), daemon=True).start()

    def _run(self, source: str, files: list) -> None:
        out = _LineWriter(self._say)
        try:
            self.home.mkdir(parents=True, exist_ok=True)
            os.environ["SENRIGAN_HOME"] = str(self.home)
            if str(_src_dir()) not in sys.path:
                sys.path.insert(0, str(_src_dir()))

            if source == "files":
                self.import_files(files)
            else:
                self.fetch_github()

            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                import ingest, train_final, check_skew  # noqa: E401 - after SENRIGAN_HOME is set

                self._stage("ingesting games")
                if ingest.main() != 0:
                    raise RuntimeError("ingest failed")

                self._stage("training model")
                candidate = self.home / "models" / (ARTIFACT_NAME + ".new")
                if train_final.main(["--out", str(candidate)]) != 0:
                    raise RuntimeError("training failed")

                self._stage("checking the new model")
                if check_skew.main(["--n", "200", "--model", str(candidate)]) != 0:
                    raise RuntimeError("the new model failed the train/serve check; kept the old one")

            final = self.home / "models" / ARTIFACT_NAME
            os.replace(candidate, final)
            self._stage("done")
            self._say(f"new ratings installed → {final.name}")
            with self._lock:
                self.done = True
        except Exception as e:  # noqa: BLE001 - shown to the user
            with self._lock:
                self.error = str(e)
            self._say(f"error: {e}")
        finally:
            with self._lock:
                self.running = False


def _sniff_year(path: Path):
    """Read the `year` column of the first rows of an unrecognised CSV."""
    try:
        import pandas as pd
        head = pd.read_csv(path, nrows=50, usecols=["year"], low_memory=False)
        y = int(head["year"].mode().iloc[0])
        return y if 2000 < y < 2100 else None
    except Exception:  # noqa: BLE001
        return None
