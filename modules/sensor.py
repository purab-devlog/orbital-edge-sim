"""
Module 4 — Sensor & Anomaly Injector
======================================
Generates synthetic hyperspectral tiles and runs a two-stage detection pipeline:

Stage 1 — Lightweight filter (always running, negligible power):
    Computes simple spectral band ratios that mimic real remote-sensing indices:
      - NBR  (Normalized Burn Ratio)       → wildfire
      - NDVI (Normalized Difference Veg.)  → crop stress / flood
      - SWIR ratio                         → pipeline leak (hydrocarbon proxy)
      - NDWI (Normalized Diff. Water)      → flood boundary
    If any index exceeds its anomaly threshold, an anomaly_score [0,1] is emitted
    and a heavy inference request is raised.

Stage 2 — Heavy inference gate (gated by power + thermal scheduler):
    When approved, "inference" runs: confidence is refined, event class confirmed,
    bounding box tightened. This is where the Sarvam Vision API would live in
    production. For the simulator we model it as a deterministic refinement step.

Radiation fault injection:
    A Poisson process (~0.5 events per 100 ticks) flips random bits in the raw
    tile band values. The lightweight filter detects implausible values (outside
    physical range [-1, 1] for indices) and flags the tile as CORRUPTED.
    Corrupted tiles are discarded — CRC detects, does not correct.

Event types:
    WILDFIRE | PIPELINE_LEAK | CROP_STRESS | FLOOD | CLEAR

Outputs per tick:
    tile            — the synthetic spectral tile (band values + metadata)
    anomaly_score   — lightweight filter score [0, 1]  (0 = clear)
    anomaly_class   — event type string
    inference_requested — bool: lightweight filter wants heavy inference
    inference_result    — dict or None: filled when heavy inference ran
    tile_corrupted      — bool: radiation fault detected
    radiation_event     — bool: a bit flip occurred this tick
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional

# ── Anomaly thresholds ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD   = 0.55     # Lightweight filter fires above this score
CONFIDENCE_FLOOR    = 0.60     # Heavy inference minimum confidence to emit event

# ── Radiation fault model ─────────────────────────────────────────────────────
# Mean rate: λ events per tick. At 30s/tick over a 94-min orbit (188 ticks),
# expect ~0.94 bit-flip events per orbit — plausible for LEO radiation environment.
RADIATION_LAMBDA    = 0.005    # Poisson rate per tick

# ── Event injection schedule ──────────────────────────────────────────────────
# (tick_start, tick_end, event_class, intensity)
# Injected deterministically so the demo always shows the same events.
INJECTED_EVENTS = [
    (10,  25, "WILDFIRE",      0.85),
    (55,  65, "PIPELINE_LEAK", 0.72),
    (100, 115,"CROP_STRESS",   0.82),   # bumped from 0.68: base band noise at ticks
                                        # 108/112 suppressed score below threshold
    (140, 155,"FLOOD",         0.80),
    (170, 180,"WILDFIRE",      0.91),
]

# Band indices used (simulated, unitless [-1, 1] after normalisation)
BANDS = ["NIR", "RED", "SWIR1", "SWIR2", "GREEN", "BLUE"]


# ── Synthetic tile ────────────────────────────────────────────────────────────
@dataclass
class SpectralTile:
    tick:        int
    lat:         float          # centre latitude  (degrees)
    lon:         float          # centre longitude (degrees)
    bands:       dict           # band_name → reflectance value [0, 1]
    event_class: str = "CLEAR"
    intensity:   float = 0.0
    corrupted:   bool = False


# ── Detection result ──────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    tick:          int
    event_class:   str
    anomaly_score: float        # lightweight filter output
    confidence:    float        # heavy inference output (0 if not run)
    bbox_lat:      tuple        # (lat_min, lat_max)
    bbox_lon:      tuple        # (lon_min, lon_max)
    inference_ran: bool
    tile_corrupted: bool
    radiation_event: bool
    inference_requested: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _normalised_index(a: float, b: float) -> float:
    """(a - b) / (a + b), returns 0 on divide-by-zero."""
    denom = a + b
    if abs(denom) < 1e-9:
        return 0.0
    return (a - b) / denom


def _base_bands(rng: random.Random) -> dict:
    """Generate a background 'clear' spectral signature."""
    nir   = rng.uniform(0.30, 0.50)
    red   = rng.uniform(0.05, 0.12)
    swir1 = rng.uniform(0.10, 0.20)
    swir2 = rng.uniform(0.08, 0.15)
    green = rng.uniform(0.08, 0.15)
    blue  = rng.uniform(0.04, 0.09)
    return {"NIR": nir, "RED": red, "SWIR1": swir1,
            "SWIR2": swir2, "GREEN": green, "BLUE": blue}


def _inject_event(bands: dict, event_class: str,
                  intensity: float, rng: random.Random) -> dict:
    """
    Modify band values to simulate a spectral event signature.
    Changes are grounded in real remote-sensing physics:
      WILDFIRE     : high SWIR (thermal emission proxy), low NIR (smoke/char)
      PIPELINE_LEAK: elevated SWIR2 (hydrocarbon absorption band ~2.3 µm)
      CROP_STRESS  : low NIR (reduced chlorophyll reflectance), high RED
      FLOOD        : high GREEN+BLUE (water), low NIR+SWIR
    """
    b = dict(bands)
    noise = rng.uniform(-0.03, 0.03)

    if event_class == "WILDFIRE":
        b["SWIR1"] = _clamp(b["SWIR1"] + 0.45 * intensity + noise)
        b["SWIR2"] = _clamp(b["SWIR2"] + 0.40 * intensity + noise)
        b["NIR"]   = _clamp(b["NIR"]   - 0.25 * intensity + noise)
        b["RED"]   = _clamp(b["RED"]   + 0.15 * intensity + noise)

    elif event_class == "PIPELINE_LEAK":
        b["SWIR2"] = _clamp(b["SWIR2"] + 0.38 * intensity + noise)
        b["SWIR1"] = _clamp(b["SWIR1"] + 0.20 * intensity + noise)

    elif event_class == "CROP_STRESS":
        b["NIR"]   = _clamp(b["NIR"]   - 0.30 * intensity + noise)
        b["RED"]   = _clamp(b["RED"]   + 0.20 * intensity + noise)

    elif event_class == "FLOOD":
        b["GREEN"] = _clamp(b["GREEN"] + 0.30 * intensity + noise)
        b["BLUE"]  = _clamp(b["BLUE"]  + 0.25 * intensity + noise)
        b["NIR"]   = _clamp(b["NIR"]   - 0.35 * intensity + noise)
        b["SWIR1"] = _clamp(b["SWIR1"] - 0.30 * intensity + noise)

    return b


def _radiation_corrupt(bands: dict, rng: random.Random) -> dict:
    """
    Flip a random bit in one band value, simulating a single-event upset.
    We model this as adding a large random offset to one band,
    producing a physically implausible value (outside [0, 1]).
    The corruption check in the lightweight filter catches this.
    """
    b = dict(bands)
    target = rng.choice(BANDS)
    # Bit flip: add ±0.5–1.5 to push value outside physical range
    b[target] = b[target] + rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
    return b


# ── Lightweight filter ────────────────────────────────────────────────────────

def _lightweight_filter(bands: dict) -> tuple[float, str, bool]:
    """
    Compute spectral indices and return (anomaly_score, event_class, corrupted).

    Corruption check: any band outside [0, 1] → CORRUPTED, score = 0.
    """
    # Physical range check (radiation fault detection)
    for v in bands.values():
        if v < -0.05 or v > 1.05:
            return 0.0, "CLEAR", True

    nir   = bands["NIR"]
    red   = bands["RED"]
    swir1 = bands["SWIR1"]
    swir2 = bands["SWIR2"]
    green = bands["GREEN"]

    # Spectral indices
    nbr  = _normalised_index(nir, swir2)    # Burn Ratio: low NBR → fire/burn
    ndvi = _normalised_index(nir, red)      # Veg Index: low NDVI → stress
    ndwi = _normalised_index(green, nir)    # Water Index: high NDWI → flood
    swir_ratio = swir2 / (swir1 + 1e-9)    # Hydrocarbon proxy: high → leak

    # Score each class
    scores = {
        "WILDFIRE":      _clamp((0.8 - nbr) * 1.2),     # low NBR → fire
        "CROP_STRESS":   _clamp((0.5 - ndvi) * 1.5),    # low NDVI → stress
        "FLOOD":         _clamp((ndwi + 0.3) * 1.1),    # high NDWI → water
        "PIPELINE_LEAK": _clamp((swir_ratio - 0.9) * 1.3),
    }

    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]

    if best_score < ANOMALY_THRESHOLD:
        return best_score, "CLEAR", False

    return best_score, best_class, False


# ── Heavy inference (simulated) ───────────────────────────────────────────────

def _heavy_inference(tile: SpectralTile,
                     anomaly_score: float,
                     anomaly_class: str) -> dict:
    """
    Simulated heavy inference step.
    In production: Sarvam Vision API call.
    Here: deterministic refinement — confidence = anomaly_score × 1.08,
    bounding box derived from tile centre.
    """
    confidence = _clamp(anomaly_score * 1.08)
    if confidence < CONFIDENCE_FLOOR:
        return {"confirmed": False, "confidence": confidence,
                "event_class": anomaly_class}

    # Bounding box: ±0.1° around tile centre (≈11 km at equator)
    bbox_lat = (round(tile.lat - 0.1, 4), round(tile.lat + 0.1, 4))
    bbox_lon = (round(tile.lon - 0.1, 4), round(tile.lon + 0.1, 4))

    return {
        "confirmed":   True,
        "confidence":  round(confidence, 3),
        "event_class": anomaly_class,
        "bbox_lat":    bbox_lat,
        "bbox_lon":    bbox_lon,
    }


# ── Main Engine ───────────────────────────────────────────────────────────────

class SensorEngine:
    """
    Call .step(tick, pos_km, inference_approved) each tick.
    Returns DetectionResult.

    Parameters
    ----------
    tick              : simulation tick (int)
    pos_km            : satellite ECI position (x, y, z) in km — used to
                        derive approximate nadir lat/lon for the tile centre
    inference_approved: bool — power AND thermal gates are both OPEN
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._tile_rng = random.Random(seed + 1)

    def step(self,
             tick: int,
             pos_km: tuple,
             inference_approved: bool) -> DetectionResult:

        # ── Derive approximate nadir coordinates from ECI position ────────
        # ECI → approximate lat/lon (ignores Earth rotation for simplicity;
        # labeled as approximate in the tile metadata)
        x, y, z = pos_km
        r = math.sqrt(x*x + y*y + z*z)
        lat = math.degrees(math.asin(_clamp(z / r, -1.0, 1.0)))
        lon = math.degrees(math.atan2(y, x))

        # ── Generate base tile ────────────────────────────────────────────
        bands = _base_bands(self._tile_rng)

        # ── Check injection schedule ──────────────────────────────────────
        event_class = "CLEAR"
        intensity   = 0.0
        for t_start, t_end, ev_class, ev_intensity in INJECTED_EVENTS:
            if t_start <= tick <= t_end:
                event_class = ev_class
                intensity   = ev_intensity
                break

        if event_class != "CLEAR":
            bands = _inject_event(bands, event_class, intensity, self._tile_rng)

        # ── Radiation fault injection ─────────────────────────────────────
        radiation_event = self._rng.random() < RADIATION_LAMBDA
        if radiation_event:
            bands = _radiation_corrupt(bands, self._rng)

        # ── Build tile ────────────────────────────────────────────────────
        tile = SpectralTile(
            tick=tick, lat=round(lat, 4), lon=round(lon, 4),
            bands=bands, event_class=event_class, intensity=intensity,
        )

        # ── Stage 1: Lightweight filter ───────────────────────────────────
        anomaly_score, detected_class, corrupted = _lightweight_filter(bands)
        tile.corrupted = corrupted

        inference_requested = (
            not corrupted and anomaly_score >= ANOMALY_THRESHOLD
        )

        # ── Stage 2: Heavy inference (if approved and requested) ──────────
        inference_result = None
        inference_ran    = False

        if inference_requested and inference_approved and not corrupted:
            inference_result = _heavy_inference(tile, anomaly_score, detected_class)
            inference_ran    = True

        # ── Build result ──────────────────────────────────────────────────
        bbox_lat = inference_result["bbox_lat"] if (inference_result and inference_result.get("confirmed")) else (round(lat-0.1,4), round(lat+0.1,4))
        bbox_lon = inference_result["bbox_lon"] if (inference_result and inference_result.get("confirmed")) else (round(lon-0.1,4), round(lon+0.1,4))

        return DetectionResult(
            tick=tick,
            event_class=detected_class if inference_ran and inference_result.get("confirmed") else event_class if not corrupted else "CLEAR",
            anomaly_score=round(anomaly_score, 4),
            confidence=round(inference_result["confidence"], 3) if inference_result else 0.0,
            bbox_lat=bbox_lat,
            bbox_lon=bbox_lon,
            inference_ran=inference_ran,
            tile_corrupted=corrupted,
            radiation_event=radiation_event,
            inference_requested=inference_requested,
        )

    def snapshot(self, result: DetectionResult) -> dict:
        return {
            "tick":                result.tick,
            "event_class":         result.event_class,
            "anomaly_score":       result.anomaly_score,
            "confidence":          result.confidence,
            "bbox_lat":            result.bbox_lat,
            "bbox_lon":            result.bbox_lon,
            "inference_ran":       result.inference_ran,
            "tile_corrupted":      result.tile_corrupted,
            "radiation_event":     result.radiation_event,
            "inference_requested": result.inference_requested,
        }
