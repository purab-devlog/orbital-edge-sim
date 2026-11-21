"""
Module 5 — Semantic Packet Serializer
=======================================
Serializes a confirmed detection event into a binary <512-byte semantic packet
for transmission over the constrained RF downlink.

This is semantic/task-oriented compression — we transmit a DECISION,
not a recoverable image. The 102,400× reduction (50 MB tile → ~490 bytes)
comes entirely from discarding raw pixel data and keeping only the
structured inference output.

Packet schema (little-endian unless noted):
┌─────────────────────────────────────────────────────────┐
│ magic         2 bytes  0x53 0x41  ('SA' ASCII)          │
│ version       1 byte   0x01                             │
│ event_class   1 byte   enum (0=CLEAR,1=WILDFIRE,…)      │
│ severity      1 byte   0–255 mapped from confidence     │
│ confidence    1 byte   0–255 (confidence × 255)         │
│ timestamp     8 bytes  Unix epoch seconds (uint64 LE)   │
│ latitude      4 bytes  IEEE 754 float32                 │
│ longitude     4 bytes  IEEE 754 float32                 │
│ bbox          16 bytes 4× float32 (lat_min,lat_max,     │
│                                    lon_min,lon_max)     │
│ flags         2 bytes  bit 0=radiation_seen,            │
│                        bit 1=inference_throttled,       │
│                        bit 2=eclipse                    │
│ payload_len   2 bytes  length of summary field (uint16) │
│ summary      variable  UTF-8 structured text, ≤466 B    │
│ crc32         4 bytes  CRC-32 of all preceding bytes    │
└─────────────────────────────────────────────────────────┘

Fixed overhead: 2+1+1+1+1+8+4+4+16+2+2 = 42 bytes  (header without CRC)
CRC field:      4 bytes
Max summary:    512 − 42 − 4 = 466 bytes  (we cap at 420 to leave headroom)
Total guaranteed ≤ 512 bytes.

Integrity:
  - Serializer computes CRC-32 and appends it.
  - Deserializer recomputes CRC-32 over [0:-4] and compares.
  - Mismatch → packet is DISCARDED (CRC detects, does not correct).
  - Corruption is simulated by flipping a random byte in the payload.
"""

import struct
import zlib
import time
from dataclasses import dataclass
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────
MAGIC            = bytes([0x53, 0x41])    # 'SA'
VERSION          = 0x01
MAX_SUMMARY_BYTES = 420
MAX_PACKET_BYTES  = 512

# Raw tile size we compare against (synthetic 50 MB hyperspectral scene)
RAW_TILE_BYTES   = 50 * 1024 * 1024      # 50 MB

# Event class encoding
EVENT_CLASSES = {
    "CLEAR":         0,
    "WILDFIRE":      1,
    "PIPELINE_LEAK": 2,
    "CROP_STRESS":   3,
    "FLOOD":         4,
}
EVENT_CLASSES_INV = {v: k for k, v in EVENT_CLASSES.items()}

# Flag bits
FLAG_RADIATION_SEEN      = 0b001
FLAG_INFERENCE_THROTTLED = 0b010
FLAG_ECLIPSE             = 0b100


# ── Packet dataclass ───────────────────────────────────────────────────────────
@dataclass
class SemanticPacket:
    # Header fields
    version:         int
    event_class:     str
    severity:        int         # 0–255
    confidence_raw:  int         # 0–255
    timestamp:       int         # Unix epoch seconds
    latitude:        float
    longitude:       float
    bbox_lat_min:    float
    bbox_lat_max:    float
    bbox_lon_min:    float
    bbox_lon_max:    float
    flags:           int
    summary:         str

    # Derived
    confidence:      float       # [0, 1]  = confidence_raw / 255
    raw_bytes:       bytes = b""
    byte_count:      int = 0
    compression_ratio: float = 0.0

    @staticmethod
    def from_detection(
        event_class:   str,
        confidence:    float,
        bbox_lat:      tuple,
        bbox_lon:      tuple,
        lat:           float,
        lon:           float,
        tick:          int,
        radiation_seen: bool = False,
        inference_throttled: bool = False,
        eclipse:       bool = False,
        base_timestamp: int = 1_700_000_000,
    ) -> "SemanticPacket":
        """Build a SemanticPacket from a DetectionResult."""
        severity_val = int(_confidence_to_severity(confidence) * 255)
        conf_raw     = int(confidence * 255)
        flags        = 0
        if radiation_seen:
            flags |= FLAG_RADIATION_SEEN
        if inference_throttled:
            flags |= FLAG_INFERENCE_THROTTLED
        if eclipse:
            flags |= FLAG_ECLIPSE

        summary = _build_summary(event_class, confidence, bbox_lat, bbox_lon,
                                 lat, lon, tick)

        pkt = SemanticPacket(
            version=VERSION,
            event_class=event_class,
            severity=severity_val,
            confidence_raw=conf_raw,
            timestamp=base_timestamp + tick * 30,
            latitude=lat,
            longitude=lon,
            bbox_lat_min=bbox_lat[0],
            bbox_lat_max=bbox_lat[1],
            bbox_lon_min=bbox_lon[0],
            bbox_lon_max=bbox_lon[1],
            flags=flags,
            summary=summary,
            confidence=round(confidence, 3),
        )
        raw = _serialize(pkt)
        pkt.raw_bytes  = raw
        pkt.byte_count = len(raw)
        pkt.compression_ratio = round(RAW_TILE_BYTES / pkt.byte_count, 1)
        return pkt


# ── Summary builder ────────────────────────────────────────────────────────────

def _confidence_to_severity(conf: float) -> float:
    """Map confidence [0,1] to severity [0,1] with a mild S-curve."""
    # Simple piecewise: < 0.7 → low, 0.7–0.85 → medium, > 0.85 → high
    if conf >= 0.85:
        return 0.95
    elif conf >= 0.70:
        return 0.70
    return 0.45


def _build_summary(event_class: str, confidence: float,
                   bbox_lat: tuple, bbox_lon: tuple,
                   lat: float, lon: float, tick: int) -> str:
    """
    Build a compact structured UTF-8 summary string.
    Format is intentionally terse to stay within byte budget.
    Example:
      EVT:WILDFIRE SEV:HIGH CONF:0.912
      LAT:12.9700 LON:77.5900
      BBOX:12.87,13.07,77.49,77.69
      T:300 ACT:EVACUATE_ZONE
    """
    sev_str = "HIGH" if confidence >= 0.85 else "MED" if confidence >= 0.70 else "LOW"
    action  = _recommended_action(event_class, confidence)
    lines = [
        f"EVT:{event_class} SEV:{sev_str} CONF:{confidence:.3f}",
        f"LAT:{lat:.4f} LON:{lon:.4f}",
        f"BBOX:{bbox_lat[0]},{bbox_lat[1]},{bbox_lon[0]},{bbox_lon[1]}",
        f"T:{tick*30} ACT:{action}",
    ]
    summary = "\n".join(lines)
    # Enforce byte cap
    encoded = summary.encode("utf-8")
    if len(encoded) > MAX_SUMMARY_BYTES:
        encoded = encoded[:MAX_SUMMARY_BYTES]
        summary = encoded.decode("utf-8", errors="ignore")
    return summary


def _recommended_action(event_class: str, confidence: float) -> str:
    actions = {
        "WILDFIRE":      "EVACUATE_ZONE"   if confidence > 0.80 else "MONITOR",
        "PIPELINE_LEAK": "ISOLATE_SEGMENT" if confidence > 0.75 else "INSPECT",
        "CROP_STRESS":   "DISPATCH_SURVEY",
        "FLOOD":         "CLOSE_ROADS"     if confidence > 0.80 else "ALERT_DISTRICT",
        "CLEAR":         "NONE",
    }
    return actions.get(event_class, "MONITOR")


# ── Serializer ─────────────────────────────────────────────────────────────────

def _serialize(pkt: SemanticPacket) -> bytes:
    """
    Pack the semantic packet into binary.
    Returns bytes with CRC-32 appended.
    """
    summary_bytes = pkt.summary.encode("utf-8")[:MAX_SUMMARY_BYTES]
    payload_len   = len(summary_bytes)

    header = struct.pack(
        "<2sBBBBQffffffHH",
        MAGIC,
        pkt.version,
        EVENT_CLASSES.get(pkt.event_class, 0),
        pkt.severity,
        pkt.confidence_raw,
        pkt.timestamp,
        pkt.latitude,
        pkt.longitude,
        pkt.bbox_lat_min,
        pkt.bbox_lat_max,
        pkt.bbox_lon_min,
        pkt.bbox_lon_max,
        pkt.flags,
        payload_len,
    )

    body    = header + summary_bytes
    crc     = zlib.crc32(body) & 0xFFFFFFFF
    crc_b   = struct.pack("<I", crc)
    return body + crc_b


# ── Deserializer ───────────────────────────────────────────────────────────────

HEADER_FMT  = "<2sBBBBQffffffHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 42 bytes


@dataclass
class ParsedPacket:
    valid:       bool
    crc_ok:      bool
    event_class: str
    severity:    int
    confidence:  float
    timestamp:   int
    latitude:    float
    longitude:   float
    bbox_lat:    tuple
    bbox_lon:    tuple
    flags:       int
    summary:     str
    byte_count:  int
    error:       str = ""


def deserialize(raw: bytes) -> ParsedPacket:
    """
    Deserialize and validate a binary packet.
    Returns ParsedPacket with valid=False and error message on any failure.
    CRC mismatch → valid=False, crc_ok=False (discard; do not attempt correction).
    """
    if len(raw) < HEADER_SIZE + 4:
        return ParsedPacket(False, False, "CLEAR", 0, 0.0, 0, 0.0, 0.0,
                            (0,0), (0,0), 0, "", len(raw),
                            error="Packet too short")

    # CRC check
    payload   = raw[:-4]
    crc_recv  = struct.unpack("<I", raw[-4:])[0]
    crc_calc  = zlib.crc32(payload) & 0xFFFFFFFF
    crc_ok    = (crc_recv == crc_calc)

    if not crc_ok:
        return ParsedPacket(False, False, "CLEAR", 0, 0.0, 0, 0.0, 0.0,
                            (0,0), (0,0), 0, "", len(raw),
                            error=f"CRC mismatch: got {crc_recv:#010x}, "
                                  f"expected {crc_calc:#010x} — packet discarded")

    # Unpack header
    try:
        fields = struct.unpack_from(HEADER_FMT, raw, 0)
    except struct.error as e:
        return ParsedPacket(False, True, "CLEAR", 0, 0.0, 0, 0.0, 0.0,
                            (0,0), (0,0), 0, "", len(raw),
                            error=f"Header unpack failed: {e}")

    magic, ver, ev_code, sev, conf_raw, ts, lat, lon, \
    blat_min, blat_max, blon_min, blon_max, flags, payload_len = fields

    if magic != MAGIC:
        return ParsedPacket(False, True, "CLEAR", 0, 0.0, 0, 0.0, 0.0,
                            (0,0), (0,0), 0, "", len(raw),
                            error=f"Bad magic: {magic!r}")

    # Extract summary
    summary_start = HEADER_SIZE
    summary_end   = summary_start + payload_len
    summary       = raw[summary_start:summary_end].decode("utf-8", errors="replace")

    return ParsedPacket(
        valid=True,
        crc_ok=True,
        event_class=EVENT_CLASSES_INV.get(ev_code, "UNKNOWN"),
        severity=sev,
        confidence=round(conf_raw / 255, 3),
        timestamp=ts,
        latitude=round(lat, 4),
        longitude=round(lon, 4),
        bbox_lat=(round(blat_min, 4), round(blat_max, 4)),
        bbox_lon=(round(blon_min, 4), round(blon_max, 4)),
        flags=flags,
        summary=summary,
        byte_count=len(raw),
    )


# ── Corruption simulator (for RF module to use) ────────────────────────────────

def corrupt_packet(raw: bytes, rng) -> bytes:
    """Flip a random byte in the packet body (not the CRC field)."""
    if len(raw) <= 4:
        return raw
    idx = rng.randint(0, len(raw) - 5)   # avoid the last 4 CRC bytes
    corrupted = bytearray(raw)
    corrupted[idx] ^= 0xFF
    return bytes(corrupted)
