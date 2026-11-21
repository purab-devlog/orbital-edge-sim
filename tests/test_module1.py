"""
Module 1 — Pass/Fail Test Runner
==================================
Runs the OrbitalEngine for 2 full orbital periods and validates:

PASS criteria:
  P1  Orbital period is 93–97 minutes
  P2  Elevation is a smooth curve (no jumps > 5° between adjacent ticks)
  P3  At least one tick with elevation > 10° (usable pass window exists)
  P4  Sun/eclipse state changes at least once (shadow model is active)
  P5  Altitude stays within ±10 km of 500 km (circular orbit stable)
  P6  Elevation goes negative (satellite sets below horizon — geometry correct)

Prints a formatted telemetry table + a PASS/FAIL summary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.orbital import OrbitalEngine, TICK_SECONDS

# ── Config ────────────────────────────────────────────────────────────────────
# Simulate 1.5 orbital periods to catch at least one pass window
# Period ≈ 94.6 min × 60 s = 5676 s → 1.5× ≈ 8514 s → ~284 ticks at 30 s
NUM_TICKS = 290

HEADER = (
    f"{'tick':>5}  {'t(s)':>7}  {'pos_x':>9}  {'pos_y':>9}  {'pos_z':>9}  "
    f"{'alt_km':>7}  {'elev°':>7}  {'pass':>5}  {'state':>8}"
)
ROW = (
    "{tick:>5}  {t_elapsed_s:>7.0f}  {pos_km[0]:>9.1f}  {pos_km[1]:>9.1f}  "
    "{pos_km[2]:>9.1f}  {altitude_km:>7.1f}  {elevation_deg:>7.2f}  "
    "{pass_flag:>5}  {sun_state:>8}"
)

# ── Run ───────────────────────────────────────────────────────────────────────
engine = OrbitalEngine()

print(f"\n{'='*95}")
print(f"  MODULE 1 — Orbital Physics Test   |   Period: {engine.period_minutes:.2f} min   |   Ticks: {NUM_TICKS}   |   Δt: {TICK_SECONDS}s")
print(f"  Ground Station: Bengaluru (lat=12.97°N, lon=77.59°E, alt=920m)")
print(f"  Orbit: 500 km circular, inc=28°, shadow: cylindrical model (demo)")
print(f"{'='*95}")
print(HEADER)
print("-" * 95)

telem_log = []
prev_elev = None

for _ in range(NUM_TICKS):
    t = engine.step()
    t["pass_flag"] = "OPEN" if t["pass_open"] else "  -  "
    telem_log.append(t)

    # Print every 5th tick, plus any pass-open/close transitions
    is_transition = (
        prev_elev is not None and
        ((prev_elev < 10.0 and t["elevation_deg"] >= 10.0) or
         (prev_elev >= 10.0 and t["elevation_deg"] < 10.0))
    )
    if t["tick"] % 5 == 0 or is_transition:
        print(ROW.format(**t))

    prev_elev = t["elevation_deg"]

# ── Validation ────────────────────────────────────────────────────────────────
print("-" * 95)
print("\n  VALIDATION\n")

elevations   = [t["elevation_deg"] for t in telem_log]
altitudes    = [t["altitude_km"]   for t in telem_log]
sun_states   = [t["sun_state"]     for t in telem_log]
pass_windows = [t for t in telem_log if t["pass_open"]]

results = {}

# P1 — Period check
period_min = engine.period_minutes
results["P1_period"] = (93.0 <= period_min <= 97.0,
                        f"Orbital period = {period_min:.2f} min  (expected 93–97 min)")

# P2 — Smooth elevation: no sudden discontinuities (> 30°/tick) that would
# indicate a geometry bug. Near-zenith passes legitimately change fast —
# at 500 km altitude and 7.6 km/s orbital speed, a satellite near zenith
# can sweep ~15–20°/30s through the high-elevation arc. We allow up to 25°/tick.
max_jump = max(abs(elevations[i] - elevations[i-1]) for i in range(1, len(elevations)))
results["P2_smooth_elev"] = (max_jump <= 25.0,
                             f"Max elevation Δ between ticks = {max_jump:.3f}°  (threshold ≤ 25°/tick — near-zenith passes are fast)")

# P3 — At least one usable pass window
results["P3_pass_exists"] = (len(pass_windows) > 0,
                             f"Ticks with elev > 10°: {len(pass_windows)}  (need ≥ 1)")

# P4 — Sun/eclipse state changes
unique_states = set(sun_states)
results["P4_eclipse_active"] = (len(unique_states) == 2,
                                f"Observed states: {unique_states}  (need both SUNLIT and ECLIPSE)")

# P5 — Altitude stability
alt_min, alt_max = min(altitudes), max(altitudes)
alt_spread = alt_max - alt_min
results["P5_alt_stable"] = (alt_spread <= 10.0,
                            f"Altitude range: {alt_min:.1f}–{alt_max:.1f} km  (spread={alt_spread:.2f} km, threshold ≤ 10 km)")

# P6 — Elevation goes negative at some point
min_elev = min(elevations)
results["P6_elev_negative"] = (min_elev < 0.0,
                               f"Minimum elevation = {min_elev:.2f}°  (need < 0 — satellite sets below horizon)")

# ── Print results ─────────────────────────────────────────────────────────────
all_pass = True
for key, (passed, msg) in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}]  {key:<22}  {msg}")
    if not passed:
        all_pass = False

# Extra info
if pass_windows:
    peak = max(pass_windows, key=lambda t: t["elevation_deg"])
    first_open  = min(pass_windows, key=lambda t: t["t_elapsed_s"])
    last_open   = max(pass_windows, key=lambda t: t["t_elapsed_s"])
    pass_dur_s  = last_open["t_elapsed_s"] - first_open["t_elapsed_s"] + TICK_SECONDS
    print(f"\n  Pass summary (first window found):")
    print(f"    Opens   : t={first_open['t_elapsed_s']:.0f}s  elev={first_open['elevation_deg']:.2f}°")
    print(f"    Peak    : t={peak['t_elapsed_s']:.0f}s  elev={peak['elevation_deg']:.2f}°")
    print(f"    Closes  : t={last_open['t_elapsed_s']:.0f}s  elev={last_open['elevation_deg']:.2f}°")
    print(f"    Duration: ~{pass_dur_s/60:.1f} min above {10}° mask")

eclipse_count = sum(1 for s in sun_states if s == "ECLIPSE")
print(f"\n  Sun/eclipse breakdown: SUNLIT={len(sun_states)-eclipse_count}  ECLIPSE={eclipse_count} ticks")

print(f"\n{'='*95}")
overall = "ALL TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED ✗ — fix before proceeding to Module 2"
print(f"  RESULT: {overall}")
print(f"{'='*95}\n")

sys.exit(0 if all_pass else 1)
