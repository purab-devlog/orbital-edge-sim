"""
Module 3 — Thermal Model
==========================
Linearised thermal model for the GPU compute node on an orbital platform.

Heat equation (Euler integration):
    T(t+1) = T(t) + [P_gpu(t) + Q_solar_abs(t) − Q_rej(t)] · Δt / C_thermal

Where:
    P_gpu(t)       = GPU heat dissipation during inference (= GPU power draw,
                     since electrical work → heat in a compute node)
    Q_solar_abs(t) = absorbed solar flux on exposed satellite surface
                     (adds heat load; zero in eclipse)
    Q_rej(t)       = radiative heat rejection (linearised Stefan-Boltzmann):
                       Q_rej = k_rad · (T − T_effective)
                     T_effective differs between sunlit and eclipse sides.

T_effective:
    In eclipse:  T_effective ≈ 50 K   (radiator sees deep space)
    In sunlight: T_effective ≈ 150 K  (radiator partly sees illuminated surfaces)
    Note: sunlight ADDS to absorbed heat via Q_solar_abs, it does not make
    rejection slower. The T_effective difference is a secondary effect.

Thermal gate thresholds:
    THROTTLE_TEMP_C  — inference throttled to 50% duty if exceeded
    CUTOFF_TEMP_C    — inference hard-cut until cooldown
    RESUME_TEMP_C    — inference may resume after a cutoff

Outputs per tick:
    gpu_temp_c         — current GPU junction temperature (°C)
    thermal_gate       — OPEN / THROTTLE / CUTOFF
    inference_allowed  — bool
    q_solar_w          — absorbed solar heat load (W)
    q_rej_w            — heat rejected this tick (W)
    p_gpu_w            — GPU heat generation this tick (W)
"""

from dataclasses import dataclass
from typing import Literal

# ── Thermal Parameters ────────────────────────────────────────────────────────
# We model the GPU junction temperature relative to the spacecraft chassis.
# The chassis itself equilibrates to roughly −30°C (eclipse) to +15°C (sunlit).
# K_RAD is the effective thermal conductance between the GPU module and the
# chassis/radiator assembly (W/K). C_THERMAL is the GPU module thermal mass.
#
# Equilibrium temperatures (sustained full inference):
#   Active + sunlit:   15 + (55+16)/0.95 ≈  90°C  → triggers cutoff
#   Active + eclipse: −30 + 55/0.95       ≈  28°C  → safe
#   Idle   + sunlit:   15 + (4+16)/0.95   ≈  36°C  → safe
#   Idle   + eclipse: −30 + 4/0.95        ≈ −26°C  → safe (cold standby)
#
# This means: the thermal gate fires ONLY during sustained inference in sunlight,
# which is exactly the duty-cycle behaviour we want to demonstrate.

C_THERMAL_J_PER_K    = 50.0    # GPU module thermal mass (J/K) — compact module
K_RAD                = 0.95    # Linearised thermal conductance GPU→chassis (W/K)

# T_effective by sun state (°C) — chassis equilibrium temperature
# Eclipse: chassis radiates to deep space, cools to ~-30°C
# Sunlit:  chassis absorbs solar flux, warms to ~+15°C
T_EFFECTIVE_ECLIPSE_C  = -30.0
T_EFFECTIVE_SUNLIT_C   = +15.0

# Solar absorbed heat load on GPU module (W) — conductive/radiative coupling
# from illuminated spacecraft body to the compute module. Small but non-zero.
Q_SOLAR_ABSORBED_W   = 16.0

# GPU heat generation (W) — same as power draw (all electrical → heat)
from modules.power import P_GPU_ACTIVE_W, P_GPU_IDLE_W

# Temperature thresholds (°C)
INITIAL_TEMP_C       = 20.0    # Starting temperature
THROTTLE_TEMP_C      = 75.0    # Begin throttling inference
CUTOFF_TEMP_C        = 85.0    # Hard stop inference
RESUME_TEMP_C        = 60.0    # Resume after cutoff (hysteresis)

ThermalGate = Literal["OPEN", "THROTTLE", "CUTOFF"]


# ── State ─────────────────────────────────────────────────────────────────────
@dataclass
class ThermalState:
    gpu_temp_c: float = INITIAL_TEMP_C
    thermal_gate: ThermalGate = "OPEN"
    inference_allowed: bool = True
    q_solar_w: float = 0.0
    q_rej_w: float = 0.0
    p_gpu_w: float = 0.0
    in_cutoff: bool = False      # True while waiting for RESUME_TEMP_C


# ── Engine ────────────────────────────────────────────────────────────────────
class ThermalEngine:
    """
    Call .step(inference_active, sun_fraction, dt_s) each tick.
    Returns updated ThermalState.
    """

    def __init__(self, initial_temp_c: float = INITIAL_TEMP_C):
        self.state = ThermalState(gpu_temp_c=initial_temp_c)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self,
             inference_active: bool,
             sun_fraction: float,
             dt_s: float = 30.0) -> ThermalState:
        """
        Advance one tick.

        Parameters
        ----------
        inference_active : GPU running inference this tick
        sun_fraction     : 0.0 (eclipse) – 1.0 (full sun)
        dt_s             : tick duration in seconds
        """
        s = self.state
        # ── Heat sources ──────────────────────────────────────────────────
        p_gpu    = P_GPU_ACTIVE_W if inference_active else P_GPU_IDLE_W
        q_solar  = Q_SOLAR_ABSORBED_W * sun_fraction
        q_in     = p_gpu + q_solar

        # ── Thermal rejection (linearised, Celsius domain) ───────────────
        # T_effective is the chassis temperature the GPU module dumps heat into.
        # Sunlight warms the chassis; eclipse lets it cool toward deep space.
        t_eff_c  = (T_EFFECTIVE_SUNLIT_C  if sun_fraction > 0
                    else T_EFFECTIVE_ECLIPSE_C)
        q_rej    = K_RAD * (s.gpu_temp_c - t_eff_c)
        q_rej    = max(0.0, q_rej)   # GPU can't absorb heat from chassis

        # ── Temperature update ────────────────────────────────────────────
        dT = (q_in - q_rej) * dt_s / C_THERMAL_J_PER_K
        new_temp_c = s.gpu_temp_c + dT

        s.gpu_temp_c = round(new_temp_c, 2)
        s.p_gpu_w    = round(p_gpu, 2)
        s.q_solar_w  = round(q_solar, 2)

        # Recompute q_rej from the POST-update temperature so the snapshot is
        # internally consistent: if gpu_temp_c > t_eff_c then q_rej > 0.
        q_rej_post = max(0.0, K_RAD * (s.gpu_temp_c - t_eff_c))
        s.q_rej_w  = round(q_rej_post, 2)

        # ── Thermal gate with hysteresis ──────────────────────────────────
        if s.in_cutoff:
            # Waiting for temperature to drop to RESUME threshold
            if s.gpu_temp_c <= RESUME_TEMP_C:
                s.in_cutoff       = False
                s.thermal_gate    = "OPEN"
                s.inference_allowed = True
            else:
                s.thermal_gate    = "CUTOFF"
                s.inference_allowed = False
        else:
            if s.gpu_temp_c >= CUTOFF_TEMP_C:
                s.in_cutoff       = True
                s.thermal_gate    = "CUTOFF"
                s.inference_allowed = False
            elif s.gpu_temp_c >= THROTTLE_TEMP_C:
                s.thermal_gate    = "THROTTLE"
                s.inference_allowed = True   # Allowed but at reduced duty
            else:
                s.thermal_gate    = "OPEN"
                s.inference_allowed = True

        return s

    def snapshot(self) -> dict:
        s = self.state
        return {
            "gpu_temp_c":        s.gpu_temp_c,
            "thermal_gate":      s.thermal_gate,
            "inference_allowed": s.inference_allowed,
            "q_solar_w":         s.q_solar_w,
            "q_rej_w":           s.q_rej_w,
            "p_gpu_w":           s.p_gpu_w,
            "in_cutoff":         s.in_cutoff,
        }
