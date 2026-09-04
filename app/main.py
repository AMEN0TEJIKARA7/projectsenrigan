"""Project Senrigan — desktop app.

A native window (pywebview) showing `app/ui/index.html`; the UI calls the
methods on `Api` below, which delegate to `engine.Predictor` — the same code
`predict.py` uses. Nothing about the model lives in the UI.

Run from a checkout:
    python app/main.py            # native window
    python app/main.py --dev      # serve the UI at http://127.0.0.1:8765 for a browser

Data folder (raw CSVs, processed parquet, model artifact) is per user:
    Windows  %LOCALAPPDATA%\\ProjectSenrigan
    macOS    ~/Library/Application Support/ProjectSenrigan
    Linux    ~/.local/share/projectsenrigan
Override with SENRIGAN_HOME. On first launch the bundled artifact is copied
there; "Update ratings" retrains into it, never into the program folder.
"""

import argparse
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_NAME = "ProjectSenrigan"
ARTIFACT_NAME = "lol_logistic_sigmoid_v1.joblib"


def bundle_root() -> Path:
    """Repository root in a checkout; the unpacked bundle in a PyInstaller build."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def default_home() -> Path:
    if os.environ.get("SENRIGAN_HOME"):
        return Path(os.environ["SENRIGAN_HOME"])
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME.lower()


HOME = default_home()
os.environ["SENRIGAN_HOME"] = str(HOME)          # before any src import
sys.path.insert(0, str(bundle_root() / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import Predictor, ResolveError  # noqa: E402
from icons import IconStore  # noqa: E402
from pipeline import Update  # noqa: E402

UI_DIR = bundle_root() / "app" / "ui"


def ensure_artifact() -> Path:
    """First launch: seed the data folder with the artifact shipped in the build."""
    dest = HOME / "models" / ARTIFACT_NAME
    if not dest.exists():
        src = bundle_root() / "models" / ARTIFACT_NAME
        if not src.exists():
            raise SystemExit(f"no model artifact at {src}; run src/train_final.py first")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return dest


class Api:
    """Methods the UI can call. Every return value is plain JSON."""

    def __init__(self):
        self._lock = threading.Lock()
        self.predictor = Predictor(ensure_artifact())
        self.update = Update(HOME)
        self.window = None
        self.icons = IconStore(HOME)
        self.icons.warm(self.predictor.lookups["champions"])

    # --- metadata / lookups -------------------------------------------------

    def meta(self) -> dict:
        m = self.predictor.meta()
        m["home"] = str(HOME)
        return m

    def search(self, kind: str, query: str, team: str = None) -> list:
        p = self.predictor
        if kind == "team":
            return p.search_teams(query or "")
        if kind == "champion":
            return p.search_champions(query or "")
        if kind == "player":
            return p.search_players(query or "", team=team)
        if kind == "league":
            return p.search_leagues(query or "")
        return []

    def champion_icons(self, names: list) -> dict:
        """name -> data URL (or null) for each champion name given."""
        return {n: self.icons.icon(n) for n in (names or []) if n}

    def team_info(self, team: str) -> dict:
        try:
            return self.predictor.team_info(team)
        except ResolveError as e:
            return {"error": str(e), "suggestions": e.suggestions}

    # --- scoring ------------------------------------------------------------

    def predict(self, payload: dict) -> dict:
        """payload: {blue, red, blue_champs?, red_champs?, blue_roster?, red_roster?,
        league?, date?, playoffs?}. Lists are 5 strings in role order or null."""
        try:
            with self._lock:
                return self.predictor.predict_dict(
                    payload.get("blue"), payload.get("red"),
                    league=payload.get("league") or None,
                    date=payload.get("date") or None,
                    playoffs=bool(payload.get("playoffs")),
                    blue_champs=_five_or_none(payload.get("blue_champs")),
                    red_champs=_five_or_none(payload.get("red_champs")),
                    blue_roster=_five_or_none(payload.get("blue_roster")),
                    red_roster=_five_or_none(payload.get("red_roster")),
                    variant=payload.get("variant") or "auto",
                    explain=True,
                )
        except ResolveError as e:
            return {"error": str(e), "kind": e.kind, "value": e.value, "suggestions": e.suggestions}
        except ValueError as e:
            return {"error": str(e)}

    # --- retraining ---------------------------------------------------------

    def start_update(self, source: str = "github") -> dict:
        self.update.run(source=source)
        return self.update.status()

    def import_csv(self) -> dict:
        """Native file picker for Oracle's Elixir CSVs, then retrain from them."""
        if self.window is None:
            return {"error": "file picker needs the native window (not available in --dev)"}
        import webview
        paths = self.window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=("Oracle's Elixir CSV (*.csv)",))
        if not paths:
            return self.update.status()
        self.update.run(source="files", files=list(paths))
        return self.update.status()

    def update_status(self) -> dict:
        st = self.update.status()
        if st["done"] and not st.get("reloaded"):
            # A finished run leaves a new artifact behind; pick it up once.
            with self._lock:
                self.predictor = Predictor(HOME / "models" / ARTIFACT_NAME)
            self.update.done = False
            st["reloaded"] = True
            st["meta"] = self.meta()
        return st

    def open_home(self) -> dict:
        import subprocess
        HOME.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(HOME))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(HOME)])
        else:
            subprocess.Popen(["xdg-open", str(HOME)])
        return {"home": str(HOME)}


def _five_or_none(v):
    if not v:
        return None
    v = [(s or "").strip() for s in v]
    return v if len(v) == 5 and all(v) else None


# --- dev mode: the same Api over HTTP so the UI can be opened in a browser ----

def serve_dev(api: Api, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            f = UI_DIR / path.lstrip("/")
            if f.is_file() and UI_DIR in f.resolve().parents:
                ctype = {"css": "text/css", "js": "text/javascript", "woff2": "font/woff2",
                         "svg": "image/svg+xml", "png": "image/png"}.get(f.suffix[1:], "application/octet-stream")
                return self._send(200, f.read_bytes(), ctype)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            name = self.path[len("/api/"):]
            n = int(self.headers.get("Content-Length") or 0)
            args = json.loads(self.rfile.read(n) or b"[]")
            fn = getattr(api, name, None)
            if fn is None or name.startswith("_"):
                return self._send(404, {"error": f"no api method {name}"})
            try:
                self._send(200, fn(*args))
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Project Senrigan dev UI at http://127.0.0.1:{port}  (data folder: {HOME})")
    srv.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="serve the UI over HTTP instead of a native window")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    api = Api()
    if args.dev:
        serve_dev(api, args.port)
        return 0

    import webview
    api.window = webview.create_window(
        "Project Senrigan", str(UI_DIR / "index.html"), js_api=api,
        width=1180, height=800, min_size=(960, 680), background_color="#0c0f13",
    )
    webview.start(debug=bool(os.environ.get("SENRIGAN_DEBUG")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
