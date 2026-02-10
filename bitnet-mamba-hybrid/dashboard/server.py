"""
BitNet-Mamba Hybrid 204M - Training Dashboard Backend
=====================================================
FastAPI server that reads training artifacts (CSV metrics, JSONL decisions,
checkpoints) and exposes them as JSON API endpoints for the frontend dashboard.

Start with:
    cd /home/lgene/meu_modelo_temp/ai-model-forge/bitnet-mamba-hybrid/dashboard
    python server.py
  or:
    uvicorn server:app --host 0.0.0.0 --port 8087 --reload
"""

import csv
import json
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
OUTPUT_DIR = Path(
    os.environ.get("TRAINING_OUTPUT_DIR", str(PROJECT_DIR / "model_204m"))
)

CSV_PATH = OUTPUT_DIR / "loss_history.csv"
JSONL_PATH = OUTPUT_DIR / "training_manager.log.jsonl"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Training constants (from project memory)
MAX_STEPS = 61_035
MAX_TOKENS = 8_000_000_000
BITLINEAR_LR_SCALE = 1.5

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="BitNet-Mamba Training Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_lan_ip() -> str:
    """Best-effort local network IP (non-loopback)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _read_csv_rows() -> list[dict]:
    """Read all rows from the loss_history CSV. Returns list of dicts."""
    if not CSV_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Skip malformed rows
                    if "step" in row and row["step"]:
                        rows.append(row)
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def _parse_csv_row(row: dict) -> dict:
    """Convert string values from CSV row into proper Python types."""
    step_str = row.get("step", "0")
    loss_str = row.get("loss", "")
    val_loss_str = row.get("val_loss", "")
    lr_str = row.get("lr", "")
    tokens_str = row.get("tokens", "")
    tps_str = row.get("tokens_per_sec", "")
    ts = row.get("timestamp", "")

    return {
        "step": int(float(step_str)) if step_str else 0,
        "loss": float(loss_str) if loss_str else None,
        "val_loss": float(val_loss_str) if val_loss_str else None,
        "lr": float(lr_str) if lr_str else None,
        "tokens": int(float(tokens_str)) if tokens_str else None,
        "tokens_per_sec": float(tps_str) if tps_str else None,
        "timestamp": ts,
    }


def _read_jsonl_entries() -> list[dict]:
    """Read all entries from the training_manager JSONL log."""
    if not JSONL_PATH.exists():
        return []
    entries: list[dict] = []
    try:
        with open(JSONL_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return entries


def _format_elapsed(seconds: float) -> str:
    """Format seconds into human-readable elapsed string."""
    if seconds < 0:
        return "0s"
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _detect_training_state(rows: list[dict]) -> str:
    """Heuristic to detect if training is running, paused, or completed."""
    if not rows:
        return "idle"
    last = _parse_csv_row(rows[-1])
    ts = last.get("timestamp", "")
    if not ts:
        return "unknown"
    try:
        last_time = datetime.fromisoformat(ts)
        age = (datetime.now() - last_time).total_seconds()
        if last["step"] and last["step"] >= MAX_STEPS:
            return "completed"
        if age < 300:
            return "running"
        elif age < 1800:
            return "paused"
        else:
            return "stopped"
    except Exception:
        return "unknown"


def _get_latest_checkpoint_name() -> Optional[str]:
    """Return filename of most recently modified checkpoint artifact."""
    candidates: list[Path] = []
    if CHECKPOINT_DIR.exists():
        try:
            candidates.extend(
                [
                    p for p in CHECKPOINT_DIR.iterdir()
                    if p.is_file() and p.suffix == ".pt"
                ]
            )
        except Exception:
            pass

    best = OUTPUT_DIR / "best_model.pt"
    if best.exists():
        candidates.append(best)

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    return latest.name


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    """Current training status summary."""
    rows = _read_csv_rows()
    if not rows:
        return {
            "step": 0,
            "max_steps": MAX_STEPS,
            "tokens": 0,
            "max_tokens": MAX_TOKENS,
            "tokens_per_sec": 0,
            "elapsed_time": "0s",
            "lr_current": 0,
            "lr_bitlinear": 0,
            "last_val_loss": None,
            "previous_val_loss": None,
            "best_val_loss": None,
            "last_step_time": None,
            "last_tokens_per_sec": None,
            "latest_checkpoint_name": None,
            "regime": "UNKNOWN",
            "state": "idle",
            "progress_pct": 0.0,
        }

    last = _parse_csv_row(rows[-1])
    first = _parse_csv_row(rows[0])

    # Find best val_loss and train loss history
    best_val = None
    loss_history: list[float] = []
    for r in rows:
        parsed = _parse_csv_row(r)
        vl = parsed.get("val_loss")
        tl = parsed.get("loss")
        if vl is not None:
            if best_val is None or vl < best_val:
                best_val = vl
        if tl is not None:
            loss_history.append(tl)

    # Status cards use latest/previous train loss values.
    last_loss = loss_history[-1] if loss_history else None
    previous_loss = loss_history[-2] if len(loss_history) > 1 else None

    # Elapsed time
    elapsed_seconds = 0.0
    try:
        t0 = datetime.fromisoformat(first["timestamp"])
        t1 = datetime.fromisoformat(last["timestamp"])
        elapsed_seconds = (t1 - t0).total_seconds()
    except Exception:
        pass

    # Regime from latest JSONL entry
    regime = "UNKNOWN"
    entries = _read_jsonl_entries()
    if entries:
        regime = entries[-1].get("regime", "UNKNOWN")

    lr_current = last.get("lr") or 0
    lr_bitlinear = lr_current * BITLINEAR_LR_SCALE

    step = last.get("step") or 0
    tokens = last.get("tokens") or 0
    progress = (step / MAX_STEPS * 100) if MAX_STEPS > 0 else 0

    state = _detect_training_state(rows)

    return {
        "step": step,
        "max_steps": MAX_STEPS,
        "tokens": tokens,
        "max_tokens": MAX_TOKENS,
        "tokens_per_sec": round(last.get("tokens_per_sec") or 0, 1),
        "elapsed_time": _format_elapsed(elapsed_seconds),
        "lr_current": lr_current,
        "lr_bitlinear": lr_bitlinear,
        "last_val_loss": round(last_loss, 6) if last_loss is not None else None,
        "previous_val_loss": round(previous_loss, 6) if previous_loss is not None else None,
        "best_val_loss": round(best_val, 6) if best_val is not None else None,
        "last_step_time": last.get("timestamp"),
        "last_tokens_per_sec": round(last.get("tokens_per_sec"), 1) if last.get("tokens_per_sec") is not None else None,
        "latest_checkpoint_name": _get_latest_checkpoint_name(),
        "regime": regime,
        "state": state,
        "progress_pct": round(progress, 2),
    }


@app.get("/api/metrics")
def get_metrics(last_n: int = Query(default=0, ge=0)):
    """Return training metrics arrays from CSV.  last_n=0 means all rows."""
    rows = _read_csv_rows()
    if last_n > 0:
        rows = rows[-last_n:]

    steps: list[int] = []
    loss: list[Optional[float]] = []
    val_loss: list[Optional[float]] = []
    lr: list[Optional[float]] = []
    tokens: list[Optional[int]] = []
    tokens_per_sec: list[Optional[float]] = []
    timestamps: list[str] = []

    for r in rows:
        p = _parse_csv_row(r)
        steps.append(p["step"])
        loss.append(p["loss"])
        val_loss.append(p["val_loss"])
        lr.append(p["lr"])
        tokens.append(p["tokens"])
        tokens_per_sec.append(p["tokens_per_sec"])
        timestamps.append(p["timestamp"])

    return {
        "steps": steps,
        "loss": loss,
        "val_loss": val_loss,
        "lr": lr,
        "tokens": tokens,
        "tokens_per_sec": tokens_per_sec,
        "timestamps": timestamps,
        "total_rows": len(steps),
    }


@app.get("/api/decisions")
def get_decisions(
    last_n: int = Query(default=50, ge=0),
    policy: str = Query(default="all"),
):
    """Return training manager events/decisions from JSONL."""
    entries = _read_jsonl_entries()

    # Build regimes timeline (transitions only)
    regimes_timeline: list[dict] = []
    for e in entries:
        regime = e.get("regime")
        step = e.get("step", 0)
        if regime and (
            not regimes_timeline or regimes_timeline[-1]["regime"] != regime
        ):
            regimes_timeline.append({"step": step, "regime": regime})

    # Filter by policy if requested
    if policy and policy != "all":
        entries = [
            e
            for e in entries
            if e.get("policy") == policy
            or (
                e.get("action")
                and e["action"].get("type")
                and policy.lower() in e["action"]["type"].lower()
            )
        ]

    total = len(entries)

    # Entries that have an action (actual decisions), used by charts/markers.
    decisions = [e for e in entries if e.get("action") is not None]
    total_decisions = len(decisions)

    if last_n > 0:
        entries = entries[-last_n:]
        decisions = decisions[-last_n:]

    return {
        "events": entries,
        "decisions": decisions,
        "total": total,
        "total_decisions": total_decisions,
        "regimes_timeline": regimes_timeline,
    }


@app.get("/api/checkpoints")
def get_checkpoints():
    """Return information about saved checkpoints."""
    checkpoints: list[dict] = []
    latest_step_metrics = None

    rows = _read_csv_rows()
    if rows:
        last = _parse_csv_row(rows[-1])
        latest_step_metrics = {
            "step": last.get("step"),
            "loss": round(last.get("loss"), 6) if last.get("loss") is not None else None,
            "lr": last.get("lr"),
            "tokens": last.get("tokens"),
            "tokens_per_sec": round(last.get("tokens_per_sec"), 1) if last.get("tokens_per_sec") is not None else None,
            "timestamp": last.get("timestamp"),
        }

    # Scan checkpoint directory
    if CHECKPOINT_DIR.exists():
        for item in sorted(CHECKPOINT_DIR.iterdir()):
            if item.is_dir():
                continue
            if not item.name.endswith(".pt"):
                continue

            stat = item.stat()
            size_mb = round(stat.st_size / (1024 * 1024), 1)
            ts = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Extract step from filename
            step = None
            match = re.search(r"(\d{5,})", item.name)
            if match:
                step = int(match.group(1))

            # Determine reason heuristic
            reason = "scheduled"
            if "best" in item.name.lower():
                reason = "best_val_loss"
            elif "interrupt" in item.name.lower():
                reason = "interrupt"
            elif "manager" in item.name.lower():
                reason = "manager_decision"

            checkpoints.append(
                {
                    "filename": item.name,
                    "step": step,
                    "size_mb": size_mb,
                    "timestamp": ts,
                    "reason": reason,
                    "_mtime_ns": stat.st_mtime_ns,
                }
            )

    # Also check for best_model.pt in output_dir root
    best = OUTPUT_DIR / "best_model.pt"
    if best.exists():
        stat = best.stat()
        checkpoints.append(
            {
                "filename": "best_model.pt",
                "step": None,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "reason": "best_val_loss",
                "_mtime_ns": stat.st_mtime_ns,
            }
        )

    # Sort by mtime descending (most recent first)
    checkpoints.sort(key=lambda c: c.get("_mtime_ns", 0), reverse=True)
    latest_checkpoint = checkpoints[0] if checkpoints else None

    for c in checkpoints:
        c.pop("_mtime_ns", None)
    if latest_checkpoint is not None:
        latest_checkpoint = dict(latest_checkpoint)
        latest_checkpoint.pop("_mtime_ns", None)

    return {
        "checkpoints": checkpoints,
        "latest_checkpoint": latest_checkpoint,
        "latest_step_metrics": latest_step_metrics,
    }


@app.get("/api/grad_norms")
def get_grad_norms():
    """Return gradient norm data from training manager JSONL observations."""
    entries = _read_jsonl_entries()

    steps: list[int] = []
    grad_norm: list[float] = []
    clipping_freq: list[float] = []

    for e in entries:
        metrics = e.get("metrics", {})
        gn = metrics.get("grad_norm")
        s = e.get("step")
        if s is not None and gn is not None:
            steps.append(s)
            grad_norm.append(gn)
            cf = metrics.get("clipping_freq")
            clipping_freq.append(cf if cf is not None else 0.0)

    return {
        "steps": steps,
        "grad_norm": grad_norm,
        "clipping_freq": clipping_freq,
    }


# ---------------------------------------------------------------------------
# Hardware monitoring helpers
# ---------------------------------------------------------------------------

def _safe_float(s: str) -> Optional[float]:
    """Parse float from nvidia-smi output, handling '[Not Supported]' etc."""
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


def _get_gpu_info() -> list[dict]:
    """Get GPU info via nvidia-smi CLI."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,"
                "utilization.gpu,power.draw,power.limit,fan.speed",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus: list[dict] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            gpus.append({
                "index": int(parts[0]) if parts[0].strip().isdigit() else 0,
                "name": parts[1],
                "temperature_c": _safe_float(parts[2]),
                "memory_used_mb": _safe_float(parts[3]),
                "memory_total_mb": _safe_float(parts[4]),
                "utilization_pct": _safe_float(parts[5]),
                "power_draw_w": _safe_float(parts[6]),
                "power_limit_w": _safe_float(parts[7]),
                "fan_speed_pct": _safe_float(parts[8]) if len(parts) > 8 else None,
            })
        return gpus
    except Exception:
        return []


def _get_cpu_info() -> dict:
    """Get CPU and RAM info via psutil."""
    if not HAS_PSUTIL:
        return {"available": False}
    mem = psutil.virtual_memory()
    info: dict = {
        "available": True,
        "cpu_percent": psutil.cpu_percent(interval=0),
        "cpu_count": psutil.cpu_count(),
        "ram_used_gb": round(mem.used / (1024 ** 3), 2),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "ram_percent": mem.percent,
    }
    # CPU temperature (best-effort)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for key in ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal", "acpitz"):
                if key in temps and temps[key]:
                    info["cpu_temp_c"] = temps[key][0].current
                    break
            if "cpu_temp_c" not in info:
                for entries in temps.values():
                    if entries:
                        info["cpu_temp_c"] = entries[0].current
                        break
    except Exception:
        pass
    return info


@app.get("/api/hardware")
def get_hardware():
    """Return current hardware metrics (GPU + CPU)."""
    return {
        "gpus": _get_gpu_info(),
        "cpu": _get_cpu_info(),
    }


# ---------------------------------------------------------------------------
# Static files & SPA fallback
# ---------------------------------------------------------------------------
STATIC_DIR = DASHBOARD_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"error": "index.html not found", "static_dir": str(STATIC_DIR)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8087"))
    lan_ip = _get_lan_ip()

    print(f"[Dashboard] Output dir : {OUTPUT_DIR}")
    print(f"[Dashboard] CSV path   : {CSV_PATH}")
    print(f"[Dashboard] JSONL path : {JSONL_PATH}")
    print(f"[Dashboard] Checkpoints: {CHECKPOINT_DIR}")
    print(f"[Dashboard] Static dir : {STATIC_DIR}")
    print(f"[Dashboard] Listening  : http://{host}:{port}")
    if host == "0.0.0.0":
        print(f"[Dashboard] LAN access : http://{lan_ip}:{port}")
    uvicorn.run(app, host=host, port=port)
