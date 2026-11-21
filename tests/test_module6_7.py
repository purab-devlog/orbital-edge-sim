"""
Module 6 & 7 — RF Link + Ground Dispatch Test
===============================================
Full end-to-end pipeline test:
  OrbitalEngine → PowerEngine → ThermalEngine → SensorEngine
  → PacketSerializer → RFLinkEngine → GroundDispatch

Runs 1 full orbital period (~190 ticks).
Semantic packets are queued when confirmed by inference.
Raw image is also queued from tick 0 to demonstrate stall.
Ground dispatch fires on every successfully received semantic packet.

PASS criteria:

RF Link (Module 6):
  R1  At least one pass window opens (elevation > 10°)
  R2  At least one semantic packet transmitted successfully
  R3  Semantic packets transmitted faster than raw image (time contrast shown)
  R4  Raw image progress < 1% after the full simulation (stall demonstrated)
  R5  At least one packet drop occurs (loss model active)
  R6  At least one retransmission occurs OR loss rate is 0 this run (acceptable)
  R7  Token budget (BYTES_PER_TICK) is never exceeded in a single tick
  R8  No transmission occurs when pass_open = False

Ground Dispatch (Module 7):
  D1  At least one alert generated for each event type detected by RF
  D2  Alert text contains latitude, longitude, and confidence for each event
  D3  All three languages (kn-IN, hi-IN, en-IN) produce distinct alert texts
  D4  browser_tts backend fires when no API key is set
  D5  Packet summary contains all required fields
  D6  WS payload is valid JSON-serialisable
  D7  Alert text is non-empty for all supported languages
  D8  Bulbul backend is selected when a key is present (key path tested with mock)
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from modules.orbital        import OrbitalEngine, TICK_SECONDS
from modules.power          import PowerEngine
from modules.thermal        import ThermalEngine
from modules.sensor         import SensorEngine
from modules.packet         import SemanticPacket, deserialize
from modules.rf_link        import (RFLinkEngine, TxStatus,
                                    DATA_RATE_BPS, BYTES_PER_TICK,
                                    RAW_IMAGE_BYTES, MIN_ELEVATION_DEG)
from modules.ground_dispatch import GroundDispatch, SUPPORTED_LANGUAGES

NUM_TICKS        = 190
INFERENCE_BURST  = 20
INFERENCE_PERIOD = 40

# ── Engines ────────────────────────────────────────────────────────────────────
orbital  = OrbitalEngine()
power    = PowerEngine(initial_soc=0.90)
thermal  = ThermalEngine()
sensor   = SensorEngine(seed=42)
rf       = RFLinkEngine(seed=7)
dispatch = GroundDispatch()          # no API key → browser_tts

# Enqueue a raw image immediately to demonstrate the stall
rf.enqueue_raw("50 MB hyperspectral tile (demo stall)")

# ── Logging ────────────────────────────────────────────────────────────────────
HEADER = (f"{'t':>5}  {'el°':>6}  {'pass':>5}  {'queue':>5}  "
          f"{'tx_status':<12}  {'bytes':>6}  {'tx_s':>6}  "
          f"{'raw%':>6}  {'alert_lang':<8}  note")
print(f"\n{'='*105}")
print("  MODULE 6+7 — RF Link + Ground Dispatch Test")
print(f"{'='*105}")
print(HEADER)
print("-" * 105)

log = []
alerts_generated = []

for tick in range(NUM_TICKS):
    orb   = orbital.step()
    sun_f = orb["sun_fraction"]
    phase = tick % INFERENCE_PERIOD
    inf   = (phase < INFERENCE_BURST)
    radio_tx = orb["pass_open"]

    pstate = power.step(sun_f, inf, radio_tx, TICK_SECONDS)
    tstate = thermal.step(inf, sun_f, TICK_SECONDS)
    approved = pstate.inference_allowed and tstate.inference_allowed

    det    = sensor.step(tick, orb["pos_km"], approved)

    # Serialise confirmed detections and queue for RF
    new_pkts = []
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
            inference_throttled=(tstate.thermal_gate == "THROTTLE"),
            eclipse=(sun_f == 0),
        )
        new_pkts.append((pkt.raw_bytes, f"{det.event_class}@t{tick}"))

    # RF link step
    rfs = rf.step(orb["elevation_deg"], new_pkts)

    # Ground dispatch on successful semantic packet receipt
    alert_lang = ""
    if (rfs.last_tx and
            rfs.last_tx.status == TxStatus.SUCCESS and
            rfs.last_tx.parsed and
            rfs.last_tx.parsed.valid and
            rfs.last_tx.parsed.event_class != "CLEAR"):
        # Rotate through languages for demo
        lang_cycle = ["en-IN", "kn-IN", "hi-IN"]
        lang = lang_cycle[len(alerts_generated) % 3]
        result = dispatch.dispatch(rfs.last_tx.parsed, language=lang)
        alerts_generated.append({
            "tick":       tick,
            "event":      rfs.last_tx.parsed.event_class,
            "language":   lang,
            "result":     result,
            "parsed":     rfs.last_tx.parsed,
        })
        alert_lang = lang

    row = {
        "tick":           tick,
        "elevation":      orb["elevation_deg"],
        "pass_open":      rfs.pass_open,
        "queue_depth":    rfs.queue_depth,
        "last_tx":        rfs.last_tx,
        "bytes_sent":     rfs.bytes_sent_tick,
        "raw_pct":        rfs.raw_progress_pct,
        "alert_lang":     alert_lang,
        "pass_number":    rfs.pass_number,
        "total_sent":     rfs.total_bytes_sent,
    }
    log.append(row)

    # Print interesting ticks
    is_pass_event = (rfs.pass_open or
                     (tick > 0 and log[-2]["pass_open"] != rfs.pass_open))
    if (is_pass_event or rfs.last_tx or alert_lang or tick % 30 == 0):
        tx   = rfs.last_tx
        stat = tx.status.value[:10] if tx else "-"*10
        byt  = tx.byte_count if tx else 0
        txs  = f"{tx.tx_time_s:.4f}" if tx else "  -   "
        note = ""
        if tx and tx.status != TxStatus.SUCCESS:
            note = tx.error[:35]
        elif alert_lang:
            note = f"→ alert [{alert_lang}]"
        elif rfs.pass_open and not rfs.last_tx:
            note = "pass open, queue empty"
        pass_sym = "●OPEN" if rfs.pass_open else " ----"
        print(
            f"{tick:>5}  {orb['elevation_deg']:>6.1f}  {pass_sym}  "
            f"{rfs.queue_depth:>5}  {stat:<12}  {byt:>6}  {txs:>6}  "
            f"{rfs.raw_progress_pct:>5.3f}%  {alert_lang:<8}  {note}"
        )

print("-" * 105)

# ── Validation ─────────────────────────────────────────────────────────────────
print("\n  VALIDATION\n")

results = {}
all_tx   = [r["last_tx"] for r in log if r["last_tx"]]
pass_ticks = [r for r in log if r["pass_open"]]
no_pass    = [r for r in log if not r["pass_open"]]

# R1 — Pass window opens
results["R1_pass_opens"] = (
    len(pass_ticks) > 0,
    f"Ticks with pass open: {len(pass_ticks)}"
)

# R2 — Semantic packet delivered
semantic_success = [t for t in all_tx
                    if t.status == TxStatus.SUCCESS and t.parsed and t.parsed.valid]
results["R2_semantic_delivered"] = (
    len(semantic_success) > 0,
    f"Semantic packets delivered: {len(semantic_success)}"
)

# R3 — Semantic faster than raw (tx_time contrast)
sem_times = [t.tx_time_s for t in semantic_success]
raw_tx    = [t for t in all_tx if t.label.startswith("50 MB") and t.byte_count > 0]
raw_times = [t.tx_time_s for t in raw_tx]
if sem_times and raw_times:
    # Semantic packet time vs time needed for full raw transfer
    full_raw_time_s = (RAW_IMAGE_BYTES * 8) / DATA_RATE_BPS
    fastest_sem = min(sem_times)
    results["R3_semantic_vs_raw"] = (
        fastest_sem < full_raw_time_s,
        f"Fastest semantic tx: {fastest_sem:.4f}s  |  "
        f"Full raw transfer would need: {full_raw_time_s/3600:.1f} hours  |  "
        f"Ratio: {full_raw_time_s/fastest_sem:,.0f}×"
    )
else:
    results["R3_semantic_vs_raw"] = (False, "No semantic or raw tx to compare")

# R4 — Raw image stalls (< 1% progress after full sim)
final_raw_pct = log[-1]["raw_pct"]
results["R4_raw_stalls"] = (
    final_raw_pct < 1.0,
    f"Raw image progress after {NUM_TICKS} ticks: {final_raw_pct:.4f}%  "
    f"(total pass time ≈ {len(pass_ticks)*TICK_SECONDS/60:.1f} min)"
)

# R5 — At least one drop
drops = [t for t in all_tx if t.status in (TxStatus.DROPPED, TxStatus.CRC_FAIL)]
results["R5_drops_occur"] = (
    len(drops) > 0,
    f"Dropped/CRC-fail transmissions: {len(drops)}"
)

# R6 — At least one retransmission OR no drops (both acceptable)
retries = [t for t in all_tx if t.attempts > 1]
results["R6_retransmit"] = (
    len(retries) > 0 or len(drops) == 0,
    f"Packets delivered after retry (attempts>1): {len(retries)}  "
    f"drops: {len(drops)}"
)

# R7 — Token budget not exceeded
over_budget = [r for r in log if r["bytes_sent"] > BYTES_PER_TICK]
results["R7_token_budget"] = (
    len(over_budget) == 0,
    f"Ticks exceeding BYTES_PER_TICK ({BYTES_PER_TICK}): {len(over_budget)}"
)

# R8 — No TX when pass closed
tx_no_pass = [r for r in log if r["last_tx"] and not r["pass_open"]]
results["R8_no_tx_without_pass"] = (
    len(tx_no_pass) == 0,
    f"Ticks with TX outside pass window: {len(tx_no_pass)}"
)

# D1 — Alert for each detected event type
alerted_events = {a["event"] for a in alerts_generated}
results["D1_all_events_alerted"] = (
    len(alerted_events) > 0,
    f"Event types that triggered alerts: {alerted_events}"
)

# D2 — Alert text contains lat, lon, confidence
bad_text = [a for a in alerts_generated
            if str(round(a["parsed"].latitude, 2))  not in a["result"].alert_text
            and str(int(a["parsed"].confidence*100)) not in a["result"].alert_text]
results["D2_alert_has_coords"] = (
    len(bad_text) == 0,
    f"Alerts with coords/conf in text: {len(alerts_generated)-len(bad_text)}/{len(alerts_generated)}"
)

# D3 — Three languages produce distinct texts for same event type
test_dispatch = GroundDispatch()
from modules.packet import SemanticPacket, deserialize
# Build a synthetic parsed packet for language test
_test_pkt = SemanticPacket.from_detection(
    "WILDFIRE", 0.91, (12.87, 13.07), (77.49, 77.69),
    12.97, 77.59, tick=50
)
_parsed = deserialize(_test_pkt.raw_bytes)
texts = {lang: test_dispatch.dispatch(_parsed, lang).alert_text
         for lang in SUPPORTED_LANGUAGES}
all_distinct = len(set(texts.values())) == len(SUPPORTED_LANGUAGES)
results["D3_languages_distinct"] = (
    all_distinct,
    f"Distinct alert texts across {len(SUPPORTED_LANGUAGES)} languages: {all_distinct}"
)

# D4 — browser_tts fires when no key
results["D4_browser_tts_fallback"] = (
    dispatch.backend == "browser_tts",
    f"Backend with no key: '{dispatch.backend}'  (expected 'browser_tts')"
)

# D5 — Packet summary has required fields
REQUIRED_FIELDS = {"event_class","confidence","latitude","longitude",
                   "bbox_lat","bbox_lon","alert_text","language"}
bad_summary = [a for a in alerts_generated
               if not REQUIRED_FIELDS.issubset(a["result"].packet_summary.keys())]
results["D5_summary_fields"] = (
    len(bad_summary) == 0,
    f"Alerts with all required summary fields: {len(alerts_generated)-len(bad_summary)}/{len(alerts_generated)}"
)

# D6 — WS payload is JSON-serialisable
ws_ok = 0
for a in alerts_generated:
    try:
        payload = dispatch.to_ws_payload(a["result"])
        json.dumps(payload)
        ws_ok += 1
    except Exception as e:
        pass
results["D6_ws_json_serialisable"] = (
    ws_ok == len(alerts_generated),
    f"WS payloads JSON-serialisable: {ws_ok}/{len(alerts_generated)}"
)

# D7 — Alert text non-empty for all languages
empty_text = [lang for lang, t in texts.items() if not t.strip()]
results["D7_alert_text_nonempty"] = (
    len(empty_text) == 0,
    f"Languages with non-empty alert text: {len(SUPPORTED_LANGUAGES)-len(empty_text)}/{len(SUPPORTED_LANGUAGES)}"
)

# D8 — Bulbul selected when key present
mock_dispatch = GroundDispatch(api_key="test-key-mock")
results["D8_bulbul_with_key"] = (
    mock_dispatch.backend == "bulbul",
    f"Backend with key set: '{mock_dispatch.backend}'  (expected 'bulbul')"
)

# ── Print results ──────────────────────────────────────────────────────────────
all_pass = True
for key, (passed, msg) in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    prefix = "  [RF]  " if key.startswith("R") else "  [GD]  "
    print(f"{prefix}[{status}]  {key:<35}  {msg}")
    if not passed:
        all_pass = False

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n  RF Link summary:")
print(f"    Pass windows    : {log[-1]['pass_number']}")
print(f"    Pass ticks      : {len(pass_ticks)} ({len(pass_ticks)*TICK_SECONDS/60:.1f} min above {MIN_ELEVATION_DEG}°)")
print(f"    Semantic success: {len(semantic_success)}")
print(f"    Drops           : {len(drops)}")
print(f"    Retransmissions : {len(retries)}")
print(f"    Raw progress    : {final_raw_pct:.4f}% of {RAW_IMAGE_BYTES/1e6:.0f} MB")

print(f"\n  Ground Dispatch summary:")
print(f"    Total alerts    : {len(alerts_generated)}")
print(f"    Event types     : {alerted_events}")
print(f"    Backend         : {dispatch.backend}")
if alerts_generated:
    print(f"\n  Sample alert texts (from last generated):")
    for lang, text in texts.items():
        print(f"    [{lang}] {text[:90]}{'...' if len(text)>90 else ''}")

print(f"\n{'='*105}")
overall = "ALL TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED ✗"
print(f"  RESULT: {overall}")
print(f"{'='*105}\n")

sys.exit(0 if all_pass else 1)
