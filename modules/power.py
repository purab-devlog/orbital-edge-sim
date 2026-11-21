"""
Module 2 — Power & Battery Model
==================================
Tick-based power budget for a solar-powered orbital node.

Physics:
    SoC(t+1) = SoC(t) + [P_solar(t)·η_charge − P_load(t)/η_discharge] · Δt / E_max

Where:
    P_solar   = panel_watts × sun_fraction × cos(incidence_angle)
                (incidence angle approximated as 0 — panels track Sun)
    P_load    = P_avionics + P_radio + P_gpu  (gpu only when inference active)
    η_charge  = charge efficiency (< 1.0, accounts for MPPT + battery losses)
    η_discharge = discharge efficiency (< 1.0, DC-DC conversion losses)
    E_max     = battery capacity in Wh

SoC floors:
    RESERVE_SOC   — hard floor for avionics + radio (always protected)
    GPU_MIN_SOC   — inference cannot start below this
    GPU_ABORT_SOC — inference aborted mid-run if SoC drops here

Outputs per tick:
    soc            — state of charge [0.0 – 1.0]
    energy_wh      — remaining energy in Wh
    p_solar_w      — instantaneous solar input (W)
    p_load_w       — instantaneous total load (W)
    p_net_w        — net power (positive = charging, negative = discharging)
    inference_allowed — bool: thermal + power gate both open
    power_gate     — string: OPEN / LOW_BATTERY / RESERVE / CHARGING_ONLY
"""

from dataclasses import dataclass, field
from typing import Literal

# ── Spacecraft Power Parameters ───────────────────────────────────────────────
PANEL_WATTS          = 120.0   # Peak solar panel output (W) — small cubesat-class
ETA_CHARGE           = 0.88    # Charge efficiency: MPPT + LiPo battery losses
ETA_DISCHARGE        = 0.92    # Discharge efficiency: DC-DC converter losses
BATTERY_CAPACITY_WH  = 80.0   # Total battery capacity (Wh)

# Load profile (W)
P_AVIONICS_W         = 8.0    # Always-on: OBC, sensors, housekeeping
P_RADIO_IDLE_W       = 3.0    # Radio in idle/receive mode
P_RADIO_TX_W         = 15.0   # Radio transmitting (RF pass open)
P_GPU_ACTIVE_W       = 55.0   # GPU during inference burst
P_GPU_IDLE_W         = 4.0    # GPU in cold-standby (clocks gated)

# SoC thresholds [0.0 – 1.0]
RESERVE_SOC          = 0.15   # Hard floor — avionics/radio always protected
GPU_MIN_SOC          = 0.35   # Inference cannot START below this
GPU_ABORT_SOC        = 0.25   # Running inference ABORTED if SoC drops here

# Solar incidence approximation
# Real satellites use articulating panels; we model cos(θ)=1.0 (best case)
# and multiply by sun_fraction from the orbital engine.
SOLAR_INCIDENCE_FACTOR = 1.0

PowerGate = Literal["OPEN", "LOW_BATTERY", "RESERVE", "CHARGING_ONLY"]


# ── State ─────────────────────────────────────────────────────────────────────
@dataclass
class PowerState:
    soc: float = 0.90                        # Initial SoC 90%
    energy_wh: float = field(init=False)
    p_solar_w: float = 0.0
    p_load_w: float = 0.0
    p_net_w: float = 0.0
    inference_running: bool = False
    radio_tx: bool = False
    power_gate: PowerGate = "OPEN"
    inference_allowed: bool = True

    def __post_init__(self):
        self.energy_wh = self.soc * BATTERY_CAPACITY_WH


# ── Engine ────────────────────────────────────────────────────────────────────
class PowerEngine:
    """
    Call .step(sun_fraction, inference_active, radio_tx, dt_s) each tick.
    Returns updated PowerState.
    """

    def __init__(self, initial_soc: float = 0.90):
        self.state = PowerState(soc=initial_soc)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self,
             sun_fraction: float,
             inference_active: bool,
             radio_tx: bool,
             dt_s: float = 30.0) -> PowerState:
        """
        Advance one tick.

        Parameters
        ----------
        sun_fraction  : 0.0 (eclipse) – 1.0 (full sun)
        inference_active : GPU running inference this tick
        radio_tx      : transmitter active this tick
        dt_s          : tick duration in seconds
        """
        s = self.state

        # ── Solar input ───────────────────────────────────────────────────
        p_solar = PANEL_WATTS * sun_fraction * SOLAR_INCIDENCE_FACTOR
        s.p_solar_w = round(p_solar, 2)

        # ── Load profile ──────────────────────────────────────────────────
        p_gpu    = P_GPU_ACTIVE_W if inference_active else P_GPU_IDLE_W
        p_radio  = P_RADIO_TX_W  if radio_tx          else P_RADIO_IDLE_W
        p_load   = P_AVIONICS_W + p_radio + p_gpu
        s.p_load_w = round(p_load, 2)
        s.inference_running = inference_active
        s.radio_tx = radio_tx

        # ── Net power (before efficiency) ─────────────────────────────────
        # Energy in: solar × charge efficiency
        # Energy out: load / discharge efficiency
        dt_h = dt_s / 3600.0
        delta_wh = (p_solar * ETA_CHARGE - p_load / ETA_DISCHARGE) * dt_h

        # ── SoC update ────────────────────────────────────────────────────
        new_energy = max(RESERVE_SOC * BATTERY_CAPACITY_WH,
                         min(BATTERY_CAPACITY_WH,
                             s.energy_wh + delta_wh))
        s.energy_wh = round(new_energy, 4)
        s.soc       = round(new_energy / BATTERY_CAPACITY_WH, 4)
        s.p_net_w   = round((delta_wh / dt_h), 2)   # instantaneous net power W

        # ── Power gate decision ───────────────────────────────────────────
        if s.soc <= RESERVE_SOC:
            s.power_gate        = "RESERVE"
            s.inference_allowed = False
        elif s.soc < GPU_MIN_SOC:
            s.power_gate        = "LOW_BATTERY"
            s.inference_allowed = False
        elif p_solar > 0 and s.soc < 0.50 and p_net_w_positive(delta_wh):
            # Preferentially charge when below 50% and in sunlight
            s.power_gate        = "CHARGING_ONLY"
            s.inference_allowed = False
        else:
            s.power_gate        = "OPEN"
            s.inference_allowed = True

        # Abort running inference if SoC dropped below abort threshold
        if inference_active and s.soc < GPU_ABORT_SOC:
            s.inference_allowed = False
            s.power_gate        = "LOW_BATTERY"

        return s

    def snapshot(self) -> dict:
        """Return current state as a plain dict for logging."""
        s = self.state
        return {
            "soc":               s.soc,
            "energy_wh":         s.energy_wh,
            "p_solar_w":         s.p_solar_w,
            "p_load_w":          s.p_load_w,
            "p_net_w":           s.p_net_w,
            "power_gate":        s.power_gate,
            "inference_allowed": s.inference_allowed,
            "inference_running": s.inference_running,
            "radio_tx":          s.radio_tx,
        }


def p_net_w_positive(delta_wh: float) -> bool:
    return delta_wh > 0
