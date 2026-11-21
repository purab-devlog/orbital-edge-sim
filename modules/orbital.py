"""
Module 1 — Orbital Physics Engine
===================================
Uses Poliastro for real two-body orbit propagation of a 500 km circular LEO.
Computes per-tick telemetry:
  - ECI position vector (km)
  - Elevation angle from Bengaluru ground station (degrees)
  - Sun/eclipse state (cylindrical shadow model)
  - Orbital period
  - Pass window open/close events (elevation > MIN_ELEVATION_DEG)

Shadow model: cylindrical approximation — satellite is in eclipse when
its ECI position falls within the shadow cylinder cast by Earth
(radius R_earth) along the Earth-Sun vector. For a demo timescale,
we use a fixed Sun direction in the orbit plane (worst-case geometry).

Elevation: computed from the LOS vector between the ground station
(ECEF → ECI approximation, Earth rotation neglected over a single pass)
and the satellite ECI position, projected onto the ground station's
local vertical (normal vector of the ellipsoid).

This is a scenario-driven pass model, not a full astrodynamics simulator.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from poliastro.util import time_range


# ── Constants ────────────────────────────────────────────────────────────────
ALTITUDE_KM       = 500.0          # Circular orbit altitude (km)
INCLINATION_DEG   = 28.0           # Inclination (degrees) — covers Bengaluru lat
RAAN_DEG          = 50.0           # Right Ascension of Ascending Node (degrees)
                                   # Tuned so the ground track passes over Bengaluru
                                   # within the first 1.5 orbital periods of the sim.
                                   # (RAAN=0° produces a ~9.4° max elevation — just below
                                   #  the 10° mask; RAAN=50° gives ~78° peak elevation.)
TICK_SECONDS      = 30             # Simulation time step (seconds)
MIN_ELEVATION_DEG = 10.0           # Minimum elevation for usable RF link (deg)

# Bengaluru ground station
GS_LAT  = 12.97   # degrees North
GS_LON  = 77.59   # degrees East
GS_ALT  = 920.0   # metres above sea level

R_EARTH_KM = 6371.0

# Sun direction: fixed unit vector in ECI (vernal equinox direction, orbit plane)
# This is a demo approximation — real Sun direction requires ephemeris.
SUN_DIRECTION = np.array([1.0, 0.0, 0.0])  # unit vector


# ── Ground Station ────────────────────────────────────────────────────────────
def _build_ground_station() -> SpheroidLocation:
    return SpheroidLocation(
        GS_LON * u.deg,
        GS_LAT * u.deg,
        GS_ALT * u.m,
        Earth,
    )


def _gs_ecef_km(gs: SpheroidLocation) -> np.ndarray:
    """Return ground station ECEF position in km."""
    coords = gs.cartesian_cords          # returns list of Quantity in metres
    return np.array([c.to_value(u.km) for c in coords])


def _gs_normal(gs: SpheroidLocation) -> np.ndarray:
    """Return the outward normal unit vector of the ellipsoid at the GS."""
    return np.array(gs.N)               # already a unit vector (dimensionless)


# ── Elevation Angle ───────────────────────────────────────────────────────────
def _elevation_deg(gs_pos_km: np.ndarray,
                   gs_normal: np.ndarray,
                   sat_pos_km: np.ndarray) -> float:
    """
    Compute elevation angle (degrees) of the satellite as seen from the GS.

    elevation = arcsin( (LOS · N̂) / |LOS| )

    where LOS = sat_pos − gs_pos, and N̂ is the local vertical unit vector.
    Returns negative values when satellite is below the horizon.
    """
    los = sat_pos_km - gs_pos_km
    los_norm = np.linalg.norm(los)
    if los_norm < 1e-9:
        return 0.0
    dot = np.dot(los, gs_normal)
    # Clamp to [-1, 1] to avoid arcsin domain errors from floating point
    sin_el = np.clip(dot / los_norm, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_el)))


# ── Eclipse Model (cylindrical shadow) ───────────────────────────────────────
def _in_eclipse(sat_pos_km: np.ndarray) -> bool:
    """
    Cylindrical shadow model.
    Satellite is in eclipse if:
      1. It is on the anti-sun side (dot product with Sun direction < 0), AND
      2. Its perpendicular distance from the Sun–Earth axis < R_earth.
    """
    # Component along Sun direction
    proj = np.dot(sat_pos_km, SUN_DIRECTION)
    if proj > 0:
        return False   # Sunlit side — always visible
    # Perpendicular distance from Sun–Earth axis
    perp = sat_pos_km - proj * SUN_DIRECTION
    dist_perp = np.linalg.norm(perp)
    return dist_perp < R_EARTH_KM


# ── Main Engine ───────────────────────────────────────────────────────────────
class OrbitalEngine:
    """
    Propagates a 500 km circular LEO orbit tick by tick.
    Call .step() to advance one TICK_SECONDS increment.
    Read .telemetry() for the current state dict.
    """

    def __init__(self, epoch: Time = None):
        self.epoch   = epoch or Time("2026-11-19 06:00:00", scale="utc")
        self._gs     = _build_ground_station()
        self._gs_pos = _gs_ecef_km(self._gs)
        self._gs_n   = _gs_normal(self._gs)
        self._orbit  = Orbit.circular(
            Earth,
            alt=ALTITUDE_KM * u.km,
            inc=INCLINATION_DEG * u.deg,
            raan=RAAN_DEG * u.deg,
            epoch=self.epoch,
        )
        self._t_elapsed_s = 0.0
        self._tick        = 0
        self._period_s    = self._orbit.period.to_value(u.s)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def period_minutes(self) -> float:
        return self._period_s / 60.0

    def step(self) -> dict:
        """Advance one tick and return telemetry dict."""
        # Propagate to current elapsed time
        orb = self._orbit.propagate(self._t_elapsed_s * u.s)

        sat_pos_km = np.array([orb.r[i].to_value(u.km) for i in range(3)])
        sat_vel_kms = np.array([orb.v[i].to_value(u.km / u.s) for i in range(3)])

        elev   = _elevation_deg(self._gs_pos, self._gs_n, sat_pos_km)
        eclipse = _in_eclipse(sat_pos_km)
        alt_km  = np.linalg.norm(sat_pos_km) - R_EARTH_KM

        tel = {
            "tick":           self._tick,
            "t_elapsed_s":    self._t_elapsed_s,
            "pos_km":         tuple(round(x, 3) for x in sat_pos_km),
            "vel_kms":        tuple(round(x, 5) for x in sat_vel_kms),
            "altitude_km":    round(alt_km, 2),
            "elevation_deg":  round(elev, 2),
            "pass_open":      elev >= MIN_ELEVATION_DEG,
            "sun_state":      "ECLIPSE" if eclipse else "SUNLIT",
            "sun_fraction":   0.0 if eclipse else 1.0,
        }

        # Advance counters for next call
        self._t_elapsed_s += TICK_SECONDS
        self._tick        += 1
        return tel

    def reset(self):
        self._t_elapsed_s = 0.0
        self._tick        = 0
