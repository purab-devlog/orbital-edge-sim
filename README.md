# Orbital Edge Intelligence Simulator

![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)
![Track](https://img.shields.io/badge/Track-Real--Time%20Voice%20%26%20Conversational%20AI-orange?style=flat-square)
![Hackathon](https://img.shields.io/badge/Hackathon-YC%20x%20Moss%20Builder%20Sprint-6C3FC4?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

A full-stack simulation of an **autonomous orbital edge inference pipeline** — from a power-and-thermally-constrained satellite GPU, through a 9.6 kbps RF downlink bottleneck, to a zero-cloud vernacular voice alert at a remote ground outpost.

Built to demonstrate the exact workflow that connects **Sarvam Vision** (on-board semantic distillation) + **hardware thermal/power gating** + **Sarvam Bulbul/Mayura** (offline vernacular dispatch) into one cohesive system — without terrestrial cloud infrastructure in the loop.

---

## What it simulates

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ORBITAL NODE (500 km LEO)                                              │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────┐  │
│  │ Poliastro │───▶│  Power   │───▶│ Thermal  │───▶│ Sensor/Anomaly  │  │
│  │  Orbit    │    │ Battery  │    │  Gate    │    │ Lightweight      │  │
│  │  Engine   │    │  SoC     │    │ GPU Temp │    │ Spectral Filter  │  │
│  └───────────┘    └──────────┘    └──────────┘    └────────┬────────┘  │
│                                                            │ if anomaly  │
│                                                   ┌────────▼────────┐  │
│                                                   │ Heavy Inference │  │
│                                                   │ (Sarvam Vision) │  │
│                                                   └────────┬────────┘  │
│                                                            │            │
│                                              ┌─────────────▼──────────┐│
│                                              │  Semantic Packet        ││
│                                              │  <512 bytes, CRC-32    ││
│                                              │  296,208× compression  ││
│                                              └─────────────┬──────────┘│
└────────────────────────────────────────────────────────────│────────────┘
                                                             │ 9.6 kbps UHF
                                              ┌──────────────▼──────────┐
                                              │  RF LINK SIMULATOR      │
                                              │  Token-bucket throttle  │
                                              │  15% packet loss + ARQ  │
                                              │  Pass window gating     │
                                              │  Raw 50 MB: stalled     │
                                              │  Semantic 162 B: 0.14s  │
                                              └──────────────┬──────────┘
                                                             │
                                              ┌──────────────▼──────────┐
                                              │  GROUND OUTPOST         │
                                              │  Sarvam Bulbul TTS  OR  │
                                              │  Browser TTS (fallback) │
                                              │  Kannada / Hindi / EN   │
                                              │  Zero cloud. Zero net.  │
                                              └─────────────────────────┘
```

### Space effects simulated

| Effect | Implementation |
|---|---|
| Orbital mechanics | Real two-body propagation via **Poliastro** (500 km circular LEO, inc=28°, RAAN=50°) |
| Eclipse / sunlight | Cylindrical shadow model; sun-fraction drives solar panel output |
| Solar power | 120 W peak panel, MPPT charge efficiency η=0.88 |
| Battery | 80 Wh LiPo, tick-based Euler integration, SoC floors for avionics and GPU |
| Thermal | Linearised GPU junction model: heat in from inference + absorbed solar, rejected to chassis |
| Thermal gating | Throttle at 75 °C, hard cutoff at 85 °C, hysteresis resume at 60 °C |
| Radiation | Poisson bit-flip events on sensor tiles; CRC-32 detects (does not correct) |
| RF link | 9,600 bps, 15% drop rate, ARQ retransmit up to 3×, priority queue |
| Pass window | Elevation-gated; typical 7–10 min above 10° mask over Bengaluru |
| Sensor data | Synthetic hyperspectral tiles with injected WILDFIRE / PIPELINE\_LEAK / CROP\_STRESS / FLOOD events |

---

## Architecture

```
orbital_edge_sim/
│
├── modules/
│   ├── orbital.py          Module 1  — Poliastro LEO orbit, elevation, eclipse
│   ├── power.py            Module 2  — Solar panel, battery SoC, power gate
│   ├── thermal.py          Module 3  — GPU thermal model, throttle/cutoff
│   ├── sensor.py           Module 4  — Synthetic tiles, spectral filter, anomaly inject
│   ├── packet.py           Module 5  — Binary <512-byte semantic packet + CRC-32
│   ├── rf_link.py          Module 6  — 9.6 kbps ARQ link, token bucket, pass window
│   └── ground_dispatch.py  Module 7  — Bulbul TTS / browser TTS, vernacular alerts
│
├── server.py               Module 8  — Flask + WebSocket, wires all 7 modules
├── dashboard.html          4-panel mission control UI (served by Flask)
│
├── tests/
│   ├── test_module1.py     Orbital physics pass/fail (6 checks)
│   ├── test_module2_3.py   Power + thermal combined (10 checks)
│   ├── test_module4_5.py   Sensor + packet serializer (14 checks)
│   └── test_module6_7.py   RF link + ground dispatch (16 checks)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Key numbers

| Metric | Value |
|---|---|
| Orbital altitude | 500 km circular |
| Orbital period | 94.62 min |
| Pass duration (Bengaluru) | ~7.5 min above 10° mask |
| Downlink rate | 9,600 bps (UHF) |
| Semantic packet size | 162–177 bytes |
| Semantic TX time | **0.135 s** |
| Raw 50 MB tile TX time | **12.1 hours** (never completes) |
| Compression ratio | **296,208×–323,635×** |
| Battery capacity | 80 Wh |
| Peak solar input | 120 W |
| GPU throttle threshold | 75 °C |
| GPU cutoff threshold | 85 °C |
| Packet loss rate | 15% |
| Max ARQ retransmits | 3 |
| Alert languages | Kannada (kn-IN), Hindi (hi-IN), English (en-IN) |

---

## Dashboard panels

**① Orbital Node — Power & Thermal**
Live orbit visualisation (canvas), battery SoC gauge, solar input, GPU temperature, power gate and thermal gate state, net power.

**② RF Bottleneck — 9.6 kbps Downlink**
Elevation angle, pass window timer, queue depth, last TX status and time, the compression contrast bar (semantic packet clears instantly; raw image stalls), sensor anomaly score, detected event class.

**③ Ground Outpost — Vernacular Dispatch**
Last alert text in the selected language, event map pins with coordinates and confidence, delivery stats (alerts, backend, dropped packets).

**④ Decision Log — Scheduler Trace**
Real-time stream of scheduler decisions:
```
t=142  Anomaly detected: WILDFIRE score=0.91
t=142  Inference denied: Thermal gate=CUTOFF GPU=85.0°C
t=151  Inference approved: SoC=61.0% GPU=68.0°C → conf=0.912
t=153  Semantic packet queued: 168B (303,057× compression)
t=160  RF pass open: elev=23.4° → WILDFIRE@t151 delivered in 0.1403s after 1 attempt(s)
t=161  Ground alert generated: kn-IN [browser_tts]
```

---

## Setup

### Requirements
- **Python 3.10** (required — poliastro's numba dependency does not support 3.11+)
- Git

### Install

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/orbital-edge-sim.git
cd orbital-edge-sim

# 2. Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install poliastro from GitHub (PyPI release uses removed `imp` module, incompatible with 3.10+)
pip install "git+https://github.com/poliastro/poliastro.git@main"

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# Then open .env and add your Sarvam API key (optional)
```

### Environment variables (`.env`)

```
# Optional — enables Sarvam Bulbul TTS
# Get your key at https://dashboard.sarvam.ai
SARVAM_API_KEY=

# Simulation speed (ms per tick, default 300ms = 100× real-time)
TICK_INTERVAL_MS=300
```

---

## Running

### Step 1 — Run all tests (must all pass before launching)

```bash
python tests/test_module1.py
python tests/test_module2_3.py
python tests/test_module4_5.py
python tests/test_module6_7.py
```

Expected output for each: `RESULT: ALL TESTS PASSED ✓`

### Step 2 — Launch the dashboard

```bash
python server.py
```

Open **http://localhost:5000** in your browser.

---

## TTS backends

| Backend | When active | Label in UI |
|---|---|---|
| **Sarvam Bulbul API** | `SARVAM_API_KEY` is set in `.env` or entered in the UI | `🔊 Bulbul API (connected)` |
| **Browser Web Speech API** | No key set | `🔉 Browser TTS (fallback — voice varies by device)` |

The key can also be entered at runtime in the dashboard without restarting the server. The browser checks `voice.localService` before claiming offline TTS capability.

---

## Semantic packet schema

```
┌──────────────────────────────────────────────────┐
│ magic         2 B   0x53 0x41 ('SA')             │
│ version       1 B   0x01                         │
│ event_class   1 B   0=CLEAR 1=WILDFIRE ...       │
│ severity      1 B   0–255                        │
│ confidence    1 B   0–255 (confidence × 255)     │
│ timestamp     8 B   Unix epoch (uint64 LE)       │
│ latitude      4 B   float32                      │
│ longitude     4 B   float32                      │
│ bbox          16 B  4× float32                   │
│ flags         2 B   radiation | throttled | ecl  │
│ payload_len   2 B   uint16                       │
│ summary       ≤420 B  UTF-8 structured text      │
│ crc32         4 B   CRC-32 of all preceding      │
└──────────────────────────────────────────────────┘
Fixed header: 42 bytes. Max total: 512 bytes.
This is semantic/task-oriented compression — a transmitted
decision, not a recoverable image. 50 MB → ~170 bytes = 296,208×.
```

---

## Extending

**Wire in real Sarvam Vision inference**
In `modules/sensor.py`, replace `_heavy_inference()` with an actual POST to the Sarvam Vision API using your developer credits. The function signature and return schema stay the same.

**Use a real TLE for the orbit**
In `modules/orbital.py`, replace `Orbit.circular()` with `Orbit.from_vectors()` using TLE-derived state vectors from `poliastro.earth.Satrec`. The elevation and eclipse logic downstream is unchanged.

**Add more event types**
In `modules/sensor.py`, extend `INJECTED_EVENTS` and add band signatures to `_inject_event()`. Add the new class to `EVENT_CLASSES` in `modules/packet.py`.

---

## Honest caveats

- The orbital model uses a fixed Sun direction (vernal equinox). A production system would use an ephemeris (e.g. `astropy.coordinates.get_body`).
- Earth's rotation is not modelled — ground station ECEF position is treated as fixed in ECI. Over one pass (~8 min) the error is <0.1°, acceptable for a demo.
- Bulbul TTS requires internet. The "zero-cloud" claim applies to the architecture target (Sarvam Edge on-device deployment), not the demo's browser TTS fallback.
- The 15-second "worst-case constrained pass" referenced in the RF link docs is a scenario parameter. Typical usable LEO passes above a 10° mask are 4–10 minutes.

---

## License

MIT
