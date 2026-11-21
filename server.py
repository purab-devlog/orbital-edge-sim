"""
server.py — Sarvam Orbital Edge Simulator
==========================================
Flask + WebSocket server that wires all 7 modules into a live simulation loop.

Endpoints:
  GET  /              → serves dashboard HTML
  GET  /api/config    → returns current sim config (speeds, thresholds, etc.)
  POST /api/key       → sets Sarvam API key at runtime (body: {"key": "..."})
  POST /api/reset     → resets simulation to t=0
  POST /api/language  → sets alert language (body: {"language": "kn-IN"})
  WS   /ws            → live telemetry stream (JSON frames at sim speed)

WebSocket message types (server → client):
  tick      — per-tick telemetry from all modules
  alert     — ground dispatch alert (with audio_b64 if Bulbul)
  log       — decision log entry
  reset     — simulation was reset
  config    — current configuration

Simulation speed:
  Real-time is 30s/tick — too slow to demo.
  Default: 1 tick every 300ms (100× speed-up).
  Configurable via TICK_INTERVAL_MS env var.
"""

import os
import json
import time
import threading
import random
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from flask_sock import Sock
from dotenv import load_dotenv

# Load .env for SARVAM_API_KEY
load_dotenv()

# ── Module imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from modules.orbital        import OrbitalEngine, TICK_SECONDS
from modules.power          import (PowerEngine, RESERVE_SOC, GPU_MIN_SOC,
                                    BATTERY_CAPACITY_WH)
from modules.thermal        import (ThermalEngine, THROTTLE_TEMP_C,
                                    CUTOFF_TEMP_C, RESUME_TEMP_C)
from modules.sensor         import SensorEngine, ANOMALY_THRESHOLD
from modules.packet         import SemanticPacket, RAW_TILE_BYTES
from modules.rf_link        import (RFLinkEngine, TxStatus,
                                    DATA_RATE_BPS, MIN_ELEVATION_DEG,
                                    RAW_IMAGE_BYTES)
from modules.ground_dispatch import GroundDispatch, SUPPORTED_LANGUAGES

# ── Config ─────────────────────────────────────────────────────────────────────
TICK_INTERVAL_MS  = int(os.environ.get("TICK_INTERVAL_MS", "300"))
INFERENCE_BURST   = 20
INFERENCE_PERIOD  = 40
NUM_TICKS_TOTAL   = 380        # 2 full orbital periods for a richer demo
LANG_CYCLE        = ["en-IN", "kn-IN", "hi-IN"]


# ── Simulation state (guarded by lock) ────────────────────────────────────────
_lock = threading.Lock()
_clients: set = set()
_sim_state: dict = {}
_running  = False
_sim_thread = None


def _make_engines(api_key=None):
    return {
        "orbital":  OrbitalEngine(),
        "power":    PowerEngine(initial_soc=0.90),
        "thermal":  ThermalEngine(),
        "sensor":   SensorEngine(seed=42),
        "rf":       RFLinkEngine(seed=7),
        "dispatch": GroundDispatch(api_key=api_key or
                                   os.environ.get("SARVAM_API_KEY", "")),
        "tick":     0,
        "alerts":   0,
        "decision_log": [],
        "api_key":  api_key or os.environ.get("SARVAM_API_KEY", ""),
        "language": "en-IN",
    }


def _reset_sim(api_key=None):
    global _sim_state
    with _lock:
        _sim_state = _make_engines(api_key)
        # Enqueue a raw image to show the stall from tick 0
        _sim_state["rf"].enqueue_raw("50 MB hyperspectral tile")
    _broadcast({"type": "reset"})


# ── Simulation loop ────────────────────────────────────────────────────────────

def _sim_loop():
    global _running
    while _running:
        t0 = time.time()
        _tick()
        elapsed = (time.time() - t0) * 1000
        sleep_ms = max(0, TICK_INTERVAL_MS - elapsed)
        time.sleep(sleep_ms / 1000)


def _log(state, msg: str):
    entry = {"t": state["tick"], "msg": msg}
    state["decision_log"].append(entry)
    if len(state["decision_log"]) > 200:
        state["decision_log"] = state["decision_log"][-200:]
    _broadcast({"type": "log", "t": state["tick"], "msg": msg})


def _tick():
    with _lock:
        s = _sim_state
        tick = s["tick"]

        if tick >= NUM_TICKS_TOTAL:
            # Loop the simulation
            new_state = _make_engines(s["api_key"])
            new_state["language"] = s["language"]
            _sim_state.update(new_state)
            _sim_state["rf"].enqueue_raw("50 MB hyperspectral tile")
            _broadcast({"type": "reset"})
            return

        # ── Module 1: Orbital ──────────────────────────────────────────
        orb   = s["orbital"].step()
        sun_f = orb["sun_fraction"]
        phase = tick % INFERENCE_PERIOD
        inf_scenario = (phase < INFERENCE_BURST)
        radio_tx = orb["pass_open"]

        # ── Module 2: Power ────────────────────────────────────────────
        pstate = s["power"].step(sun_f, inf_scenario, radio_tx, TICK_SECONDS)

        # ── Module 3: Thermal ──────────────────────────────────────────
        tstate = s["thermal"].step(inf_scenario, sun_f, TICK_SECONDS)

        # Gate decisions
        p_gate = pstate.power_gate
        t_gate = tstate.thermal_gate
        approved = pstate.inference_allowed and tstate.inference_allowed

        if inf_scenario and not approved:
            reason = []
            if not pstate.inference_allowed:
                reason.append(f"Power gate={p_gate} SoC={pstate.soc*100:.1f}%")
            if not tstate.inference_allowed:
                reason.append(f"Thermal gate={t_gate} GPU={tstate.gpu_temp_c:.1f}°C")
            _log(s, f"Inference denied: {' | '.join(reason)}")

        # ── Module 4: Sensor ───────────────────────────────────────────
        det = s["sensor"].step(tick, orb["pos_km"], approved)

        if det.radiation_event:
            _log(s, f"⚡ Radiation bit-flip on tile (tick={tick})")
        if det.tile_corrupted:
            _log(s, f"Tile corrupted — discarded (CRC would fail)")
        if det.anomaly_score >= ANOMALY_THRESHOLD and not det.tile_corrupted:
            _log(s, f"Anomaly detected: {det.event_class} score={det.anomaly_score:.2f}")
        if det.inference_ran:
            _log(s, f"Inference approved: SoC={pstate.soc*100:.1f}% GPU={tstate.gpu_temp_c:.1f}°C → conf={det.confidence:.3f}")

        # ── Module 5: Packet ───────────────────────────────────────────
        new_pkts = []
        pkt_meta = None
        if det.inference_ran and det.confidence >= 0.60 and det.event_class != "CLEAR":
            pkt = SemanticPacket.from_detection(
                event_class=det.event_class,
                confidence=det.confidence,
                bbox_lat=det.bbox_lat,
                bbox_lon=det.bbox_lon,
                lat=round(orb["pos_km"][0] / 100, 4),
                lon=round(orb["pos_km"][1] / 100, 4),
                tick=tick,
                radiation_seen=det.radiation_event,
                inference_throttled=(t_gate == "THROTTLE"),
                eclipse=(sun_f == 0),
            )
            new_pkts.append((pkt.raw_bytes, f"{det.event_class}@t{tick}"))
            pkt_meta = {
                "byte_count":        pkt.byte_count,
                "compression_ratio": pkt.compression_ratio,
                "event_class":       pkt.event_class,
                "confidence":        pkt.confidence,
            }
            _log(s, f"Semantic packet queued: {pkt.byte_count}B "
                    f"({pkt.compression_ratio:,.0f}× compression)")

        # ── Module 6: RF Link ──────────────────────────────────────────
        rfs = s["rf"].step(orb["elevation_deg"], new_pkts)

        if rfs.pass_open and rfs.last_tx:
            tx = rfs.last_tx
            if tx.status == TxStatus.SUCCESS and tx.parsed and tx.parsed.valid:
                _log(s, f"RF pass open: elev={rfs.elevation_deg:.1f}° "
                        f"→ {tx.label} delivered in {tx.tx_time_s:.4f}s "
                        f"after {tx.attempts} attempt(s)")
            elif tx.status in (TxStatus.DROPPED, TxStatus.CRC_FAIL):
                _log(s, f"RF drop: {tx.error}")
            elif tx.status == TxStatus.MAX_RETRY:
                _log(s, f"RF max retry exceeded: {tx.label} discarded")

        # ── Module 7: Ground Dispatch ──────────────────────────────────
        alert_payload = None
        if (rfs.last_tx and
                rfs.last_tx.status == TxStatus.SUCCESS and
                rfs.last_tx.parsed and
                rfs.last_tx.parsed.valid and
                rfs.last_tx.parsed.event_class != "CLEAR"):

            lang   = LANG_CYCLE[s["alerts"] % 3] if s["language"] == "auto" \
                     else s["language"]
            result = s["dispatch"].dispatch(rfs.last_tx.parsed, language=lang)
            s["alerts"] += 1

            _log(s, f"Ground alert generated: {lang} [{result.backend_used}]")

            alert_payload = s["dispatch"].to_ws_payload(result)
            alert_payload.pop("type", None)   # remove "ground_alert" so our "alert" wins
            alert_payload["tick"] = tick
            _broadcast({"type": "alert", **alert_payload})

        # ── Build tick telemetry ───────────────────────────────────────
        tx = rfs.last_tx
        frame = {
            "type": "tick",
            "t":    tick,
            # Orbital
            "pos_km":       orb["pos_km"],
            "elevation_deg":orb["elevation_deg"],
            "altitude_km":  orb["altitude_km"],
            "sun_state":    orb["sun_state"],
            "sun_fraction": sun_f,
            "pass_open":    orb["pass_open"],
            # Power
            "soc":          pstate.soc,
            "energy_wh":    pstate.energy_wh,
            "p_solar_w":    pstate.p_solar_w,
            "p_load_w":     pstate.p_load_w,
            "p_net_w":      pstate.p_net_w,
            "power_gate":   p_gate,
            # Thermal
            "gpu_temp_c":   tstate.gpu_temp_c,
            "thermal_gate": t_gate,
            "q_rej_w":      tstate.q_rej_w,
            "p_gpu_w":      tstate.p_gpu_w,
            # Sensor
            "anomaly_score":    det.anomaly_score,
            "event_class":      det.event_class,
            "inference_ran":    det.inference_ran,
            "confidence":       det.confidence,
            "tile_corrupted":   det.tile_corrupted,
            "radiation_event":  det.radiation_event,
            # Packet
            "pkt_meta":     pkt_meta,
            # RF
            "rf_pass_open":     rfs.pass_open,
            "rf_elevation":     rfs.elevation_deg,
            "rf_queue_depth":   rfs.queue_depth,
            "rf_raw_pct":       rfs.raw_progress_pct,
            "rf_bytes_sent":    rfs.bytes_sent_tick,
            "rf_pass_number":   rfs.pass_number,
            "rf_window_s":      rfs.window_elapsed_s,
            "rf_total_sent":    rfs.total_bytes_sent,
            "rf_tx_status":     tx.status.value if tx else None,
            "rf_tx_bytes":      tx.byte_count   if tx else 0,
            "rf_tx_time_s":     tx.tx_time_s    if tx else 0,
            "rf_tx_label":      tx.label        if tx else "",
            "rf_tx_error":      tx.error        if tx else "",
            "rf_tx_attempts":   tx.attempts     if tx else 0,
            # Ground
            "alert_payload":    alert_payload,
            # Thresholds (for gauge rendering)
            "thresh_throttle":  THROTTLE_TEMP_C,
            "thresh_cutoff":    CUTOFF_TEMP_C,
            "thresh_soc_min":   RESERVE_SOC,
            "thresh_soc_gpu":   GPU_MIN_SOC,
            "battery_cap_wh":   BATTERY_CAPACITY_WH,
            "raw_image_bytes":  RAW_IMAGE_BYTES,
        }

        s["tick"] = tick + 1
        _broadcast(frame)


# ── WebSocket broadcast ────────────────────────────────────────────────────────

def _broadcast(msg: dict):
    dead = set()
    payload = json.dumps(msg, default=str)
    for ws in list(_clients):
        try:
            ws.send(payload)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ── Flask app ──────────────────────────────────────────────────────────────────
app  = Flask(__name__)
sock = Sock(app)

DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


@app.route("/")
def index():
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@app.route("/api/config")
def config():
    with _lock:
        s = _sim_state
        return jsonify({
            "tick_interval_ms": TICK_INTERVAL_MS,
            "num_ticks":        NUM_TICKS_TOTAL,
            "data_rate_bps":    DATA_RATE_BPS,
            "min_elevation":    MIN_ELEVATION_DEG,
            "raw_image_bytes":  RAW_IMAGE_BYTES,
            "raw_tile_bytes":   RAW_TILE_BYTES,
            "language":         s.get("language", "en-IN"),
            "backend":          s["dispatch"].backend,
            "has_api_key":      bool(s.get("api_key")),
            "supported_languages": SUPPORTED_LANGUAGES,
        })


@app.route("/api/key", methods=["POST"])
def set_key():
    key = request.json.get("key", "").strip()
    with _lock:
        _sim_state["api_key"] = key
        _sim_state["dispatch"].set_api_key(key)
    return jsonify({"backend": _sim_state["dispatch"].backend,
                    "has_key": bool(key)})


@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        key = _sim_state.get("api_key", "")
        lang = _sim_state.get("language", "en-IN")
    _reset_sim(api_key=key)
    with _lock:
        _sim_state["language"] = lang
    return jsonify({"status": "reset"})


@app.route("/api/language", methods=["POST"])
def set_language():
    lang = request.json.get("language", "en-IN")
    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({"error": "Unsupported language"}), 400
    with _lock:
        _sim_state["language"] = lang
    return jsonify({"language": lang})


@sock.route("/ws")
def websocket(ws):
    _clients.add(ws)
    # Send current config immediately on connect
    with _lock:
        s = _sim_state
        ws.send(json.dumps({
            "type":    "config",
            "backend": s["dispatch"].backend,
            "has_key": bool(s.get("api_key")),
            "language": s.get("language", "en-IN"),
        }))
    try:
        while True:
            msg = ws.receive(timeout=60)
            if msg is None:
                break
            # Handle client messages (key set, language change, reset)
            try:
                data = json.loads(msg)
                if data.get("type") == "set_key":
                    with _lock:
                        _sim_state["api_key"] = data.get("key", "")
                        _sim_state["dispatch"].set_api_key(data.get("key", ""))
                    ws.send(json.dumps({
                        "type": "config",
                        "backend": _sim_state["dispatch"].backend,
                        "has_key": bool(_sim_state.get("api_key")),
                    }))
                elif data.get("type") == "set_language":
                    lang = data.get("language", "en-IN")
                    if lang in SUPPORTED_LANGUAGES:
                        with _lock:
                            _sim_state["language"] = lang
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _reset_sim()
    _running = True
    _sim_thread = threading.Thread(target=_sim_loop, daemon=True)
    _sim_thread.start()
    print(f"\n  Sarvam Orbital Edge Simulator")
    print(f"  ─────────────────────────────")
    print(f"  Dashboard : http://localhost:5000")
    print(f"  API key   : {'SET (Bulbul)' if os.environ.get('SARVAM_API_KEY') else 'NOT SET (browser TTS fallback)'}")
    print(f"  Tick speed: {TICK_INTERVAL_MS}ms per tick (1 tick = 30s simulated)\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
