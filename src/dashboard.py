"""
dashboard.py
============
Browser-based training dashboard for HammerTime. Launches a local web server
and opens the dashboard in your default browser. Training metrics stream live
via Server-Sent Events.

Usage:

    python src/dashboard.py --mode binary --target_class hammer
    python src/dashboard.py --mode multi
    python src/dashboard.py --watch_only   # attach to existing training run
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
except ImportError:
    print("[ERR] pip install fastapi uvicorn")
    sys.exit(1)

ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_defaults(mode: str) -> dict:
    script = "src/binary/train_b.py" if mode == "binary" else "src/multi/train_m.py"
    path   = ROOT / script
    spec   = importlib.util.spec_from_file_location("train_mod", path)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# CSV / stdout parsing
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(csv_path: Path) -> list[dict]:
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append({k.strip(): v.strip() for k, v in row.items()})
    except Exception:
        pass
    return rows


def extract_metrics(row: dict) -> dict | None:
    def _f(*keys):
        for k in keys:
            v = row.get(k)
            if v:
                try: return float(v)
                except ValueError: pass
        return None

    epoch = _f("epoch")
    if epoch is None:
        return None
    return {
        "epoch":    int(epoch),
        "map50":    _f("metrics/mAP50(B)", "metrics/mAP_0.5"),
        "map95":    _f("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"),
        "prec":     _f("metrics/precision(B)"),
        "recall":   _f("metrics/recall(B)"),
        "box_loss": _f("val/box_loss"),
        "cls_loss": _f("val/cls_loss"),
        "dfl_loss": _f("val/dfl_loss"),
    }


ANSI_RE  = re.compile(r"\x1b\[[0-9;]*m|\x1b\[[0-9;]*[A-Za-z]")
RATIO_RE = re.compile(r"(\d+)/(\d+)")


def parse_batch(line: str) -> tuple[int, int] | None:
    if "640:" not in line:
        return None
    clean   = ANSI_RE.sub("", line)
    clean   = clean.encode("ascii", errors="ignore").decode("ascii")
    matches = RATIO_RE.findall(clean)
    if len(matches) >= 2:
        b, total = int(matches[-1][0]), int(matches[-1][1])
        if total < 500:
            return b, total
    return None


def find_csv(mode: str, start_time: float) -> Path | None:
    base = ROOT / "runs" / mode
    if not base.exists():
        return None
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if run.stat().st_mtime < start_time:
            continue
        c = run / "results.csv"
        if c.exists():
            return c
    return None


def fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    h, rem = divmod(int(seconds), 3600)
    m, _   = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

class TrainingState:
    def __init__(self, total_epochs: int, patience: int):
        self.total_epochs  = total_epochs
        self.patience      = patience
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 46
        self.history       : list[dict] = []
        self.epoch_times   : list[float] = []
        self.start_time    = time.time()
        self.best          : dict = {}
        self.done          = False
        self.lock          = threading.Lock()

    def to_json(self) -> str:
        with self.lock:
            elapsed  = time.time() - self.start_time
            ep_pct   = self.current_epoch / self.total_epochs if self.total_epochs else 0

            if self.current_epoch >= 25 and len(self.epoch_times) >= 10:
                avg       = sum(self.epoch_times[-10:]) / 10
                rem_secs  = avg * (self.total_epochs - self.current_epoch)
                remaining = fmt_time(rem_secs)
            else:
                remaining = None

            return json.dumps({
                "epoch":        self.current_epoch,
                "total_epochs": self.total_epochs,
                "patience":     self.patience,
                "epoch_pct":    ep_pct,
                "elapsed":      fmt_time(elapsed),
                "remaining":    remaining,
                "history":      self.history,
                "best":         self.best,
                "done":         self.done,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Stdout reader thread
# ─────────────────────────────────────────────────────────────────────────────

def stdout_reader(proc, state, mode, start_time):
    csv_path      = None
    last_epoch    = 0
    last_epoch_ts = time.time()

    for raw in proc.stdout:
        lines = raw.decode("utf-8", errors="replace").split("\r")
        for line in lines:
            result = parse_batch(line)
            if result:
                b, total = result
                with state.lock:
                    state.current_batch = b
                    state.total_batches = total

            if csv_path is None:
                csv_path = find_csv(mode, start_time)

            if csv_path and csv_path.exists():
                rows    = read_csv(csv_path)
                history = [r for r in (extract_metrics(row) for row in rows) if r]
                if history:
                    with state.lock:
                        state.history = history
                        current = history[-1]["epoch"]
                        if current > last_epoch:
                            now = time.time()
                            if last_epoch > 0:
                                state.epoch_times.append(now - last_epoch_ts)
                            last_epoch_ts = now
                            last_epoch    = current
                            state.current_epoch = current
                            best = max(
                                (h for h in history if h["map50"] is not None),
                                key=lambda h: h["map50"],
                                default=history[-1],
                            )
                            state.best = best

    with state.lock:
        state.done = True


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HammerTime — Training Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {
    --bg: #0f0f0f;
    --surface: #161616;
    --border: #252525;
    --text: #e2e2e2;
    --muted: #666;
    --blue: #378ADD;
    --green: #1D9E75;
    --red: #D85A30;
    --purple: #7F77DD;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    padding: 20px;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
  }
  header h1 { font-size: 13px; font-weight: 500; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: blink 2s infinite; }
  .dot.done { background: var(--muted); animation: none; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

  .row { display: grid; gap: 12px; margin-bottom: 12px; }
  .row-2 { grid-template-columns: 1fr 1fr; }
  .row-1 { grid-template-columns: 1fr; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px 20px; }
  .card-label { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }

  /* Runtime */
  .runtime-row { display: flex; align-items: center; gap: 20px; }
  .ring-container { position: relative; width: 88px; height: 88px; flex-shrink: 0; }
  .ring-container svg { width: 88px; height: 88px; transform: rotate(-90deg); }
  .ring-bg  { fill: none; stroke: #232323; stroke-width: 8; }
  .ring-arc { fill: none; stroke: var(--blue); stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset .8s cubic-bezier(.4,0,.2,1); }
  .ring-label {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    text-align: center; pointer-events: none;
  }
  .ring-epoch { font-size: 15px; font-weight: 600; line-height: 1; }
  .ring-sep { display: block; width: 20px; height: 1px; background: #333; margin: 4px auto; }
  .ring-total { font-size: 11px; color: var(--muted); }
  .runtime-text { flex: 1; }
  .runtime-elapsed { font-size: 26px; font-weight: 600; letter-spacing: -.01em; margin-bottom: 4px; }
  .runtime-remaining { font-size: 11px; color: var(--muted); }

  /* Legend */
  .legend-items { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; }
  .legend-item { display: flex; gap: 10px; align-items: flex-start; }
  .legend-bar { width: 3px; height: 36px; border-radius: 2px; flex-shrink: 0; margin-top: 1px; }
  .legend-name { font-size: 11px; font-weight: 600; margin-bottom: 2px; }
  .legend-desc { font-size: 10px; color: var(--muted); line-height: 1.45; }

  /* Charts */
  .chart-wrap { height: 260px; position: relative; }

  /* Best epoch */
  .best-row { display: flex; align-items: center; gap: 0; }
  .best-num { font-size: 36px; font-weight: 700; color: var(--blue); padding-right: 24px; margin-right: 24px; border-right: 1px solid var(--border); }
  .best-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; flex: 1; }
  .best-metric { padding: 0 20px; border-right: 1px solid var(--border); }
  .best-metric:last-child { border-right: none; }
  .best-metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }
  .best-metric-value { font-size: 20px; font-weight: 600; }
</style>
</head>
<body>

<header>
  <div class="dot" id="statusDot"></div>
  <h1>HammerTime &mdash; Training Dashboard</h1>
</header>

<div class="row row-2">
  <div class="card">
    <div class="card-label">Runtime</div>
    <div class="runtime-row">
      <div class="ring-container">
        <svg viewBox="0 0 88 88">
          <circle class="ring-bg" cx="44" cy="44" r="36"/>
          <circle class="ring-arc" id="ringArc" cx="44" cy="44" r="36"
            stroke-dasharray="226.2" stroke-dashoffset="226.2"/>
        </svg>
        <div class="ring-label">
          <div class="ring-epoch" id="ringEpoch">0</div>
          <span class="ring-sep"></span>
          <div class="ring-total" id="ringTotal">400</div>
        </div>
      </div>
      <div class="runtime-text">
        <div class="runtime-elapsed" id="elapsed">0m</div>
        <div class="runtime-remaining" id="remaining">Estimate available after epoch 25</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-label">Legend</div>
    <div class="legend-items">
      <div class="legend-item">
        <div class="legend-bar" style="background:var(--blue)"></div>
        <div><div class="legend-name" style="color:var(--blue)">mAP@50</div><div class="legend-desc">Detection accuracy<br>at IoU 0.50</div></div>
      </div>
      <div class="legend-item">
        <div class="legend-bar" style="background:var(--green)"></div>
        <div><div class="legend-name" style="color:var(--green)">mAP@50-95</div><div class="legend-desc">Accuracy across<br>IoU thresholds</div></div>
      </div>
      <div class="legend-item">
        <div class="legend-bar" style="background:var(--red)"></div>
        <div><div class="legend-name" style="color:var(--red)">Precision</div><div class="legend-desc">True positives /<br>all detections</div></div>
      </div>
      <div class="legend-item">
        <div class="legend-bar" style="background:var(--purple)"></div>
        <div><div class="legend-name" style="color:var(--purple)">Recall</div><div class="legend-desc">True positives /<br>all ground truth</div></div>
      </div>
    </div>
  </div>
</div>

<div class="row row-2">
  <div class="card">
    <div class="card-label">All Epochs</div>
    <div class="chart-wrap">
      <canvas id="chartFull" role="img" aria-label="All epochs metrics">Metrics over all epochs.</canvas>
    </div>
  </div>
  <div class="card">
    <div class="card-label" id="patienceLabel">Patience Window</div>
    <div class="chart-wrap">
      <canvas id="chartRecent" role="img" aria-label="Patience window metrics">Metrics for patience window.</canvas>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-label">Best Epoch</div>
  <div class="best-row">
    <div class="best-num" id="bestEpoch">—</div>
    <div class="best-metrics">
      <div class="best-metric">
        <div class="best-metric-label">mAP@50</div>
        <div class="best-metric-value" style="color:var(--blue)" id="bestMap50">—</div>
      </div>
      <div class="best-metric">
        <div class="best-metric-label">mAP@50-95</div>
        <div class="best-metric-value" style="color:var(--green)" id="bestMap95">—</div>
      </div>
      <div class="best-metric">
        <div class="best-metric-label">Precision</div>
        <div class="best-metric-value" style="color:var(--red)" id="bestPrec">—</div>
      </div>
      <div class="best-metric">
        <div class="best-metric-label">Recall</div>
        <div class="best-metric-value" style="color:var(--purple)" id="bestRecall">—</div>
      </div>
    </div>
  </div>
</div>

<script>
const CIRC = 2 * Math.PI * 36;

const sharedOpts = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: '#444', font: { size: 10, family: 'monospace' }, maxTicksLimit: 8 }, grid: { color: '#1d1d1d' } },
    y: { min: 0, max: 1, ticks: { color: '#444', font: { size: 10 }, stepSize: 0.25 }, grid: { color: '#1d1d1d' } }
  }
};

function makeDatasets(history) {
  const cfg = [
    { key: 'map50',  color: '#378ADD', dash: [] },
    { key: 'map95',  color: '#1D9E75', dash: [5,3] },
    { key: 'prec',   color: '#D85A30', dash: [] },
    { key: 'recall', color: '#7F77DD', dash: [5,3] },
  ];
  return {
    labels: history.map(h => h.epoch),
    datasets: cfg.map(c => ({
      data: history.map(h => h[c.key] ?? null),
      borderColor: c.color,
      borderDash: c.dash,
      borderWidth: 1.8,
      pointRadius: history.length < 10 ? 3 : 0,
      pointBackgroundColor: c.color,
      tension: 0.3,
    }))
  };
}

const chartFull   = new Chart(document.getElementById('chartFull'),   { type: 'line', data: { labels: [], datasets: [] }, options: sharedOpts });
const chartRecent = new Chart(document.getElementById('chartRecent'), { type: 'line', data: { labels: [], datasets: [] }, options: sharedOpts });

function updateChart(chart, history) {
  const d = makeDatasets(history);
  chart.data.labels   = d.labels;
  chart.data.datasets = d.datasets;
  chart.update('none');
}

function fmt(v) { return v != null ? v.toFixed(4) : '—'; }

function applyState(d) {
  const offset = CIRC * (1 - d.epoch_pct);
  document.getElementById('ringArc').style.strokeDashoffset = offset;
  document.getElementById('ringEpoch').textContent  = d.epoch;
  document.getElementById('ringTotal').textContent  = d.total_epochs;
  document.getElementById('elapsed').textContent    = d.elapsed;
  document.getElementById('remaining').textContent  = d.remaining ?? 'Estimate available after epoch 25';
  document.getElementById('patienceLabel').textContent = `Patience Window: ${d.patience} epochs`;

  if (d.history.length >= 2) {
    updateChart(chartFull, d.history);
    updateChart(chartRecent, d.history.slice(-d.patience));
  }

  if (d.best && d.best.epoch) {
    document.getElementById('bestEpoch').textContent  = d.best.epoch;
    document.getElementById('bestMap50').textContent  = fmt(d.best.map50);
    document.getElementById('bestMap95').textContent  = fmt(d.best.map95);
    document.getElementById('bestPrec').textContent   = fmt(d.best.prec);
    document.getElementById('bestRecall').textContent = fmt(d.best.recall);
  }

  if (d.done) {
    document.getElementById('statusDot').classList.add('done');
  }
}

const es = new EventSource('/stream');
es.onmessage = e => { try { applyState(JSON.parse(e.data)); } catch(err) { console.error(err); } };
es.onerror   = () => { document.getElementById('remaining').textContent = 'Connection lost — reload to reconnect'; };
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app   = FastAPI()
STATE : TrainingState | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD_HTML


@app.get("/stream")
def stream():
    def gen():
        while True:
            if STATE is not None:
                yield f"data: {STATE.to_json()}\n\n"
            time.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/state")
def state_endpoint():
    return json.loads(STATE.to_json()) if STATE else {}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HammerTime browser dashboard")
    p.add_argument("--mode",         default="binary", choices=["binary", "multi"])
    p.add_argument("--target_class", default=None)
    p.add_argument("--watch_only",   action="store_true")
    p.add_argument("--public",       action="store_true", help="Expose via ngrok for remote access")
    p.add_argument("--port",         type=int, default=7842)
    return p.parse_args()


def main():
    global STATE
    args         = parse_args()
    defaults     = load_defaults(args.mode)
    total_epochs = defaults.get("epochs", 400)
    patience     = defaults.get("patience", 50)

    STATE      = TrainingState(total_epochs, patience)
    start_time = time.time()

    if not args.watch_only:
        train_script = (
            "src/binary/train_b.py" if args.mode == "binary"
            else "src/multi/train_m.py"
        )
        train_proc = subprocess.Popen(
            [sys.executable, str(ROOT / train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        threading.Thread(
            target=stdout_reader,
            args=(train_proc, STATE, args.mode, start_time),
            daemon=True,
        ).start()
    else:
        def watch_existing():
            csv_path      = None
            last_epoch    = 0
            last_epoch_ts = time.time()
            while True:
                if csv_path is None:
                    csv_path = find_csv(args.mode, 0)
                if csv_path and csv_path.exists():
                    rows    = read_csv(csv_path)
                    history = [r for r in (extract_metrics(row) for row in rows) if r]
                    if history:
                        with STATE.lock:
                            STATE.history = history
                            current = history[-1]["epoch"]
                            if current > last_epoch:
                                now = time.time()
                                if last_epoch > 0:
                                    STATE.epoch_times.append(now - last_epoch_ts)
                                last_epoch_ts = now
                                last_epoch    = current
                                STATE.current_epoch = current
                                best = max(
                                    (h for h in history if h["map50"] is not None),
                                    key=lambda h: h["map50"],
                                    default=history[-1],
                                )
                                STATE.best = best
                time.sleep(3)

        threading.Thread(target=watch_existing, daemon=True).start()

    url = f"http://localhost:{args.port}"
    print(f"\n  HammerTime Dashboard → {url}")

    if args.public:
        def launch_ngrok():
            time.sleep(2)
            try:
                result = subprocess.run(
                    ["ngrok", "http", str(args.port), "--log=false"],
                    capture_output=True, text=True, timeout=5
                )
            except FileNotFoundError:
                print("  [ngrok] not found -- install from https://ngrok.com/download")
                return
            except subprocess.TimeoutExpired:
                pass

            # Get public URL from ngrok API
            time.sleep(1.5)
            try:
                import urllib.request, json as _json
                with urllib.request.urlopen("http://localhost:4040/api/tunnels") as r:
                    tunnels = _json.loads(r.read())
                    public_url = tunnels["tunnels"][0]["public_url"]
                    print(f"  Public URL → {public_url}")
                    webbrowser.open(public_url)
            except Exception:
                print("  [ngrok] tunnel started -- check http://localhost:4040 for public URL")

        def start_ngrok():
            time.sleep(2)
            try:
                ngrok_proc = subprocess.Popen(
                    ["ngrok", "http", str(args.port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2)
                import urllib.request, json as _json
                with urllib.request.urlopen("http://localhost:4040/api/tunnels") as r:
                    tunnels = _json.loads(r.read())
                    public_url = tunnels["tunnels"][0]["public_url"]
                    print(f"  Public URL  → {public_url}")
                    webbrowser.open(public_url)
            except FileNotFoundError:
                print("  [ngrok] not found -- install from https://ngrok.com/download")
            except Exception as e:
                print(f"  [ngrok] error: {e} -- check http://localhost:4040 for public URL")

        threading.Thread(target=start_ngrok, daemon=True).start()
        print("  Launching ngrok for public access...")
    else:
        print("  Press Ctrl+C to stop.\n")
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
