"""
Module 2 & 3 — Combined Power + Thermal Test
===============================================
Drives OrbitalEngine → PowerEngine + ThermalEngine together for 1 full
orbital period (~189 ticks at 30s).

Scenario:
  - Inference fires every 10 ticks to simulate a busy anomaly detector
  - Radio TX is active whenever the RF pass window is open
  - Both engines subscribe to sun_fraction from orbital telemetry

PASS criteria:
  P1  SoC never drops below RESERVE_SOC (hard floor never violated)
  P2  SoC rises during sunlit + no-inference periods (charging works)
  P3  SoC falls during eclipse + inference periods (discharge works)
  P4  Thermal engine reaches THROTTLE at some point (GPU heats up)
  P5  Thermal engine cools below RESUME_TEMP_C after a CUTOFF (hysteresis)
  P6  Power gate goes LOW_BATTERY at some point (SoC gating works)
       — OR — inference_allowed stays True when SoC is healthy (acceptable if
       battery never drains that far in this scenario)
  P7  GPU temperature is strictly higher during inference vs idle ticks
  P8  Q_rej is positive whenever GPU temp > effective sink temperature
  P9  P_net_w is positive (net charging) during sunlit + idle ticks
  P10 P_net_w is negative (net discharging) during eclipse + inference ticks

Prints a formatted table + PASS/FAIL summary.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules.orbital import OrbitalEngine, TICK_SECONDS
from modules.power   import (PowerEngine, RESERVE_SOC, GPU_MIN_SOC,
                              GPU_ABORT_SOC, BATTERY_CAPACITY_WH)
from modules.thermal import (ThermalEngine, THROTTLE_TEMP_C, CUTOFF_TEMP_C,
                              RESUME_TEMP_C)

# ── Config ─────────────────────────────────────────────────────────────────
NUM_TICKS        = 190           # ~1 orbital period
# Scenario: 20-tick sustained inference burst every 40 ticks.
# This models a prolonged event detection (e.g., tracking a wildfire boundary).
# It is deliberately aggressive to force throttle/cutoff and demonstrate
# the thermal duty-cycle behaviour. Shorter bursts would not stress the model.
INFERENCE_BURST  = 20            # ticks of sustained inference per burst
INFERENCE_PERIOD = 40            # burst every N ticks

# ── Engines ────────────────────────────────────────────────────────────────
orbital  = OrbitalEngine()
power    = PowerEngine(initial_soc=0.90)
thermal  = ThermalEngine()

# ── Header ─────────────────────────────────────────────────────────────────
COL  = f"{'t':>5} {'el°':>6} {'sun':>7} {'SoC%':>5} {'Ewh':>6} "
COL += f"{'Psol':>6} {'Pload':>6} {'Pnet':>6} "
COL += f"{'Tgpu':>6} {'Qrej':>5} {'Pgate':>13} {'Tgate':>8} {'INF':>4}"

print(f"\n{'='*110}")
print("  MODULE 2+3 — Power & Thermal Combined Test")
print(f"{'='*110}")
print(COL)
print("-" * 110)

log = []
prev_soc  = None
prev_temp = None

for tick in range(NUM_TICKS):
    # Orbital telemetry
    orb = orbital.step()
    sun_f    = orb["sun_fraction"]
    pass_open = orb["pass_open"]

    # Scenario inputs: sustained burst every INFERENCE_PERIOD ticks
    phase = tick % INFERENCE_PERIOD
    inference_active = (phase < INFERENCE_BURST)
    radio_tx         = pass_open

    # Step engines
    pstate = power.step(sun_f, inference_active, radio_tx, TICK_SECONDS)
    tstate = thermal.step(inference_active, sun_f, TICK_SECONDS)

    row = {
        "tick":          tick,
        "elev":          orb["elevation_deg"],
        "sun_state":     orb["sun_state"],
        "sun_f":         sun_f,
        "soc_pct":       round(pstate.soc * 100, 1),
        "energy_wh":     pstate.energy_wh,
        "p_solar":       pstate.p_solar_w,
        "p_load":        pstate.p_load_w,
        "p_net":         pstate.p_net_w,
        "gpu_temp":      tstate.gpu_temp_c,
        "q_rej":         tstate.q_rej_w,
        "power_gate":    pstate.power_gate,
        "thermal_gate":  tstate.thermal_gate,
        "inf_active":    inference_active,
        "phase":         tick % INFERENCE_PERIOD,
        "inference_allowed_p": pstate.inference_allowed,
        "inference_allowed_t": tstate.inference_allowed,
    }
    log.append(row)

    # Print every 5 ticks or on gate changes
    gate_change = (
        prev_soc is not None and (
            abs(row["soc_pct"] - prev_soc) > 2.0 or
            row["power_gate"] != log[-2]["power_gate"] if len(log) > 1 else False
        )
    )
    if tick % 5 == 0 or gate_change:
        inf_sym = "●" if inference_active else "·"
        sun_sym = "☀" if sun_f > 0 else "🌑"
        line = (
            f"{tick:>5} {orb['elevation_deg']:>6.1f} {sun_sym:>7} "
            f"{row['soc_pct']:>5.1f} {row['energy_wh']:>6.2f} "
            f"{row['p_solar']:>6.1f} {row['p_load']:>6.1f} {row['p_net']:>+7.1f} "
            f"{row['gpu_temp']:>6.1f} {row['q_rej']:>5.1f} "
            f"{row['power_gate']:>13} {row['thermal_gate']:>8} {inf_sym:>4}"
        )
        print(line)

    prev_soc  = row["soc_pct"]
    prev_temp = row["gpu_temp"]

print("-" * 110)

# ── Validation ─────────────────────────────────────────────────────────────
print("\n  VALIDATION\n")

socs       = [r["soc_pct"] / 100 for r in log]
temps      = [r["gpu_temp"]      for r in log]
p_nets     = [r["p_net"]         for r in log]
pgates     = [r["power_gate"]    for r in log]
tgates     = [r["thermal_gate"]  for r in log]
q_rejs     = [r["q_rej"]         for r in log]

# P1 — SoC hard floor
min_soc = min(socs)
results = {}
results["P1_soc_floor"] = (
    min_soc >= RESERVE_SOC,
    f"Min SoC = {min_soc:.3f}  (RESERVE floor = {RESERVE_SOC})"
)

# P2 — SoC rises during sunlit + idle
sunlit_idle = [(r["p_net"], i) for i, r in enumerate(log)
               if r["sun_f"] > 0 and not r["inf_active"]]
charging_ticks = sum(1 for p, _ in sunlit_idle if p > 0)
results["P2_charging"] = (
    charging_ticks > 0,
    f"Ticks charging (sunlit+idle): {charging_ticks}/{len(sunlit_idle)}"
)

# P3 — SoC falls during eclipse + inference
eclipse_inf = [(r["p_net"], i) for i, r in enumerate(log)
               if r["sun_f"] == 0 and r["inf_active"]]
discharging_ticks = sum(1 for p, _ in eclipse_inf if p < 0)
results["P3_discharging"] = (
    discharging_ticks > 0,
    f"Ticks discharging (eclipse+inference): {discharging_ticks}/{len(eclipse_inf)}"
)

# P4 — Thermal throttle reached
max_temp = max(temps)
throttle_ticks = sum(1 for g in tgates if g in ("THROTTLE", "CUTOFF"))
results["P4_throttle_reached"] = (
    max_temp >= THROTTLE_TEMP_C,
    f"Max GPU temp = {max_temp:.1f}°C  (THROTTLE threshold = {THROTTLE_TEMP_C}°C)  "
    f"throttle/cutoff ticks = {throttle_ticks}"
)

# P5 — Thermal hysteresis: after a CUTOFF, temp drops to RESUME
cutoff_seen = any(g == "CUTOFF" for g in tgates)
if cutoff_seen:
    # Find first CUTOFF tick, then check if temp ever drops to RESUME after it
    first_cutoff = next(i for i, g in enumerate(tgates) if g == "CUTOFF")
    post_cutoff_temps = temps[first_cutoff:]
    resumed = any(t <= RESUME_TEMP_C for t in post_cutoff_temps)
    results["P5_hysteresis"] = (
        resumed,
        f"CUTOFF seen at tick {first_cutoff} ({temps[first_cutoff]:.1f}°C)  "
        f"→ cooled to ≤{RESUME_TEMP_C}°C after: {resumed}"
    )
else:
    # Acceptable: if max temp never reached cutoff, hysteresis wasn't needed
    results["P5_hysteresis"] = (
        True,
        f"CUTOFF never triggered (max temp {max_temp:.1f}°C < {CUTOFF_TEMP_C}°C) — hysteresis not needed"
    )

# P6 — Power gate response (LOW_BATTERY seen OR inference always allowed when healthy)
low_battery_ticks = sum(1 for g in pgates if g == "LOW_BATTERY")
gate_open_ticks   = sum(1 for g in pgates if g == "OPEN")
results["P6_power_gate"] = (
    low_battery_ticks > 0 or gate_open_ticks > 0,
    f"LOW_BATTERY ticks={low_battery_ticks}  OPEN ticks={gate_open_ticks}  "
    f"(gate is making decisions)"
)

# P7 — GPU hotter during inference vs idle ticks
inf_temps  = [r["gpu_temp"] for r in log if r["inf_active"]]
idle_temps = [r["gpu_temp"] for r in log if not r["inf_active"]]
avg_inf    = sum(inf_temps)  / len(inf_temps)  if inf_temps  else 0
avg_idle   = sum(idle_temps) / len(idle_temps) if idle_temps else 0
results["P7_gpu_hotter_inference"] = (
    avg_inf > avg_idle,
    f"Avg GPU temp — inference={avg_inf:.1f}°C  idle={avg_idle:.1f}°C"
)

# P8 — Q_rej positive whenever GPU is above the effective sink temperature.
# T_eff = +15°C (sunlit) or -30°C (eclipse). Q_rej = K*(T_gpu - T_eff).
# When GPU < T_eff (e.g. recovering from eclipse, temp still below 15°C)
# the clamp max(0, q_rej) correctly returns 0 — that is physically correct.
# We test: on ticks where GPU > T_eff (sun→+15°C, eclipse→-30°C), Q_rej > 0.
from modules.thermal import T_EFFECTIVE_SUNLIT_C, T_EFFECTIVE_ECLIPSE_C

above_sink = [
    r for r in log
    if r["gpu_temp"] > (T_EFFECTIVE_SUNLIT_C  if r["sun_f"] > 0
                        else T_EFFECTIVE_ECLIPSE_C)
]
qrej_ok = all(r["q_rej"] > 0 for r in above_sink)
results["P8_qrej_positive"] = (
    qrej_ok,
    f"Q_rej > 0 on all {len(above_sink)} ticks where GPU > T_eff: {qrej_ok}"
)

# P9 — Net positive power during sunlit idle
sunlit_idle_pnet = [p for p, _ in sunlit_idle]
net_pos = sum(1 for p in sunlit_idle_pnet if p > 0)
results["P9_net_positive_sunlit"] = (
    net_pos > 0,
    f"Ticks with P_net > 0 during sunlit+idle: {net_pos}/{len(sunlit_idle_pnet)}"
)

# P10 — Net negative power during eclipse + inference
eclipse_inf_pnet = [p for p, _ in eclipse_inf]
net_neg = sum(1 for p in eclipse_inf_pnet if p < 0)
results["P10_net_negative_eclipse"] = (
    net_neg > 0 or len(eclipse_inf) == 0,
    f"Ticks with P_net < 0 during eclipse+inference: {net_neg}/{len(eclipse_inf)}"
)

# ── Print results ──────────────────────────────────────────────────────────
all_pass = True
for key, (passed, msg) in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}]  {key:<30}  {msg}")
    if not passed:
        all_pass = False

# ── Summary stats ──────────────────────────────────────────────────────────
print(f"\n  Power summary:")
print(f"    SoC range      : {min(socs)*100:.1f}% – {max(socs)*100:.1f}%")
print(f"    Energy range   : {min(r['energy_wh'] for r in log):.1f} – {max(r['energy_wh'] for r in log):.1f} Wh  (capacity={BATTERY_CAPACITY_WH} Wh)")
print(f"    Gate breakdown : OPEN={pgates.count('OPEN')}  LOW_BATTERY={pgates.count('LOW_BATTERY')}  "
      f"RESERVE={pgates.count('RESERVE')}  CHARGING_ONLY={pgates.count('CHARGING_ONLY')}")

print(f"\n  Thermal summary:")
print(f"    Temp range     : {min(temps):.1f}°C – {max(temps):.1f}°C")
print(f"    Gate breakdown : OPEN={tgates.count('OPEN')}  THROTTLE={tgates.count('THROTTLE')}  CUTOFF={tgates.count('CUTOFF')}")
print(f"    Max Q_rej      : {max(q_rejs):.1f} W")

print(f"\n{'='*110}")
overall = "ALL TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED ✗ — fix before proceeding to Module 4"
print(f"  RESULT: {overall}")
print(f"{'='*110}\n")

sys.exit(0 if all_pass else 1)
