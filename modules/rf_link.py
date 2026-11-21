"""
Module 6 — RF Link Simulator
==============================
Simulates a constrained UHF/S-band downlink from an orbital node to a fixed
ground station. Models:

  - Pass window gating: link only open when elevation > MIN_ELEVATION_DEG
  - Token-bucket throughput: 9,600 bps enforced per tick
  - Random packet loss: 15% drop probability per transmission attempt
  - CRC validation on receipt (delegates to packet.deserialize)
  - ARQ (Automatic Repeat reQuest): NACK → retransmit, up to MAX_RETRIES
  - Priority queue: semantic packets ahead of raw imagery in the queue
  - Raw image transfer simulation: shows how a 50 MB image stalls

Terminology used in comments:
  "frame"  — one tick (30 seconds of simulated time)
  "pass"   — continuous window where elevation > MIN_ELEVATION_DEG
  "packet" — a serialized SemanticPacket binary blob (<512 bytes)

Key numbers (labeled in the UI for the demo):
  Worst-case short pass : 15 s above mask (configurable scenario)
  Typical pass budget   : 9,600 bps ÷ 8 × T_pass_s bytes
  Semantic packet       : ~162–177 bytes → clears in ~0.14 s at 9.6 kbps
  Raw 50 MB image       : would need ~11.6 hours at 9.6 kbps — never transfers

Output per tick (RFLinkState):
  pass_open          — bool: elevation above mask
  elevation_deg      — current elevation angle
  window_elapsed_s   — seconds into current pass window
  bytes_sent_tick    — bytes actually transmitted this tick
  bytes_budget_tick  — token bucket budget this tick (bytes available)
  queue_depth        — packets waiting in priority queue
  last_tx            — most recent TransmitResult (or None)
  raw_progress_pct   — simulated raw image transfer progress [0–100]
  total_bytes_sent   — cumulative bytes sent this pass
  pass_number        — how many passes have opened so far
"""

import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from modules.packet import deserialize, ParsedPacket, MAX_PACKET_BYTES

# ── RF Parameters ─────────────────────────────────────────────────────────────
DATA_RATE_BPS        = 9_600          # UHF downlink data rate (bits per second)
DATA_RATE_BYTES_S    = DATA_RATE_BPS / 8   # 1,200 bytes/second
TICK_S               = 30.0          # seconds per simulation tick
BYTES_PER_TICK       = int(DATA_RATE_BYTES_S * TICK_S)   # 36,000 bytes/tick budget
                                     # (but a semantic packet is ~170 bytes,
                                     #  so we transmit in <<1 s of wall time)

PACKET_LOSS_RATE     = 0.15          # 15% drop probability per attempt
MAX_RETRIES          = 3             # Maximum ARQ retransmit attempts
MIN_ELEVATION_DEG    = 10.0          # Minimum elevation for usable link

# Raw image stall demo
RAW_IMAGE_BYTES      = 50 * 1024 * 1024    # 50 MB synthetic tile
RAW_IMAGE_LABEL      = "50 MB hyperspectral tile"

# Queue priority levels (lower = higher priority)
PRIORITY_SEMANTIC    = 0
PRIORITY_RAW         = 1


class TxStatus(str, Enum):
    SUCCESS      = "SUCCESS"
    DROPPED      = "DROPPED"       # Random link loss
    CRC_FAIL     = "CRC_FAIL"      # CRC mismatch on receive
    MAX_RETRY    = "MAX_RETRY"     # Gave up after MAX_RETRIES
    NO_WINDOW    = "NO_WINDOW"     # Pass not open
    QUEUED       = "QUEUED"        # Added to queue, not yet sent


@dataclass
class QueueItem:
    priority:    int               # 0 = semantic, 1 = raw
    raw_bytes:   bytes
    label:       str               # human-readable description
    retries:     int = 0


@dataclass
class TransmitResult:
    status:         TxStatus
    label:          str
    byte_count:     int
    attempts:       int
    tx_time_s:      float          # simulated transmission time in seconds
    elevation_deg:  float
    parsed:         Optional[ParsedPacket] = None
    error:          str = ""


@dataclass
class RFLinkState:
    pass_open:         bool = False
    elevation_deg:     float = 0.0
    window_elapsed_s:  float = 0.0
    bytes_sent_tick:   int = 0
    bytes_budget_tick: int = 0
    queue_depth:       int = 0
    last_tx:           Optional[TransmitResult] = None
    raw_progress_pct:  float = 0.0
    total_bytes_sent:  int = 0
    pass_number:       int = 0
    token_bucket:      int = 0     # bytes remaining in bucket this tick


# ── Engine ─────────────────────────────────────────────────────────────────────
class RFLinkEngine:
    """
    Call .step(elevation_deg, packets_to_queue) each tick.

    Parameters
    ----------
    elevation_deg    : current satellite elevation angle (degrees)
    packets_to_queue : list of (raw_bytes, label) tuples to enqueue this tick.
                       Semantic packets are always enqueued at PRIORITY_SEMANTIC.

    Returns RFLinkState for this tick.
    """

    def __init__(self, seed: int = 7):
        self._rng          = random.Random(seed)
        self._queue: List[QueueItem] = []
        self._state        = RFLinkState()
        self._pass_was_open = False
        self._raw_sent_bytes = 0        # tracks raw image stall simulation
        self._window_start_tick = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def enqueue_semantic(self, raw_bytes: bytes, label: str):
        """Add a semantic packet to the high-priority queue."""
        self._queue.append(QueueItem(PRIORITY_SEMANTIC, raw_bytes, label))
        self._queue.sort(key=lambda x: x.priority)

    def enqueue_raw(self, label: str = RAW_IMAGE_LABEL):
        """
        Simulate enqueuing a raw image. We don't store 50 MB in memory —
        we track it as a progress counter.
        """
        # Only one raw image in the queue at a time
        already = any(q.priority == PRIORITY_RAW for q in self._queue)
        if not already:
            # Sentinel: raw_bytes is empty; engine uses _raw_sent_bytes counter
            self._queue.append(QueueItem(PRIORITY_RAW, b"", label))
            self._queue.sort(key=lambda x: x.priority)

    def step(self,
             elevation_deg: float,
             packets_to_queue: Optional[list] = None) -> RFLinkState:
        """Advance one tick."""
        s = self._state

        # Enqueue new packets provided this tick
        if packets_to_queue:
            for raw_bytes, label in packets_to_queue:
                self.enqueue_semantic(raw_bytes, label)

        s.elevation_deg    = round(elevation_deg, 2)
        s.pass_open        = elevation_deg >= MIN_ELEVATION_DEG
        s.queue_depth      = len(self._queue)
        s.bytes_budget_tick = BYTES_PER_TICK
        s.bytes_sent_tick  = 0
        s.last_tx          = None

        # Pass window bookkeeping
        if s.pass_open and not self._pass_was_open:
            s.pass_number      += 1
            s.window_elapsed_s  = 0.0
            s.total_bytes_sent  = 0
        if s.pass_open:
            s.window_elapsed_s += TICK_S
        else:
            s.window_elapsed_s  = 0.0

        self._pass_was_open = s.pass_open

        # Token bucket: replenish each tick
        s.token_bucket = BYTES_PER_TICK

        # Only transmit when pass is open
        if not s.pass_open:
            s.raw_progress_pct = self._raw_progress_pct()
            return s

        # Process semantic packets first (all that fit in budget this tick),
        # then one raw chunk with whatever budget remains.
        # A 162-byte semantic packet uses 0.45% of the 36,000-byte tick budget,
        # so we can drain the entire semantic backlog in one tick if needed.
        last_semantic_tx = None

        # Drain semantic queue within budget
        while self._queue and s.token_bucket > 0:
            item = self._queue[0]

            if item.priority != PRIORITY_SEMANTIC:
                break  # hit the raw sentinel — stop semantic drain

            if len(item.raw_bytes) > s.token_bucket:
                break  # packet doesn't fit in remaining budget

            result = self._transmit_semantic(item, s)
            s.bytes_sent_tick  += result.byte_count
            s.total_bytes_sent += result.byte_count
            s.token_bucket     -= result.byte_count

            if result.status == TxStatus.SUCCESS:
                self._queue.pop(0)
                last_semantic_tx = result
            elif result.status == TxStatus.MAX_RETRY:
                self._queue.pop(0)   # discard after max retries
                last_semantic_tx = result
            else:
                # DROPPED or CRC_FAIL: item stays for retry next tick
                last_semantic_tx = result
                break  # don't retry the same item twice in one tick

        # If a semantic packet was attempted this tick, report it
        if last_semantic_tx is not None:
            s.last_tx = last_semantic_tx
        else:
            # No semantic packets — try one raw chunk
            raw_items = [q for q in self._queue if q.priority == PRIORITY_RAW]
            if raw_items and s.token_bucket > 0:
                result = self._transmit_raw_chunk(s)
                s.last_tx          = result
                s.bytes_sent_tick  += result.byte_count
                s.total_bytes_sent += result.byte_count
                if self._raw_sent_bytes >= RAW_IMAGE_BYTES:
                    # Remove raw sentinel (never actually reached in demo)
                    self._queue = [q for q in self._queue
                                   if q.priority != PRIORITY_RAW]

        s.raw_progress_pct = self._raw_progress_pct()
        return s

    # ── Internal transmission logic ────────────────────────────────────────────

    def _transmit_semantic(self,
                           item: QueueItem,
                           s: RFLinkState) -> TransmitResult:
        """
        Attempt to transmit one semantic packet with ARQ.
        Returns the result of this tick's attempt.
        """
        byte_count = len(item.raw_bytes)
        tx_time_s  = (byte_count * 8) / DATA_RATE_BPS

        item.retries += 1
        attempt = item.retries

        # Simulate random link loss
        if self._rng.random() < PACKET_LOSS_RATE:
            if attempt >= MAX_RETRIES:
                return TransmitResult(
                    TxStatus.MAX_RETRY, item.label, byte_count, attempt,
                    tx_time_s, s.elevation_deg,
                    error=f"Dropped {MAX_RETRIES} times — discarded"
                )
            return TransmitResult(
                TxStatus.DROPPED, item.label, byte_count, attempt,
                tx_time_s, s.elevation_deg,
                error=f"Link drop #{attempt} — retransmitting"
            )

        # Simulate CRC check on received packet
        parsed = deserialize(item.raw_bytes)
        if not parsed.crc_ok:
            if attempt >= MAX_RETRIES:
                return TransmitResult(
                    TxStatus.MAX_RETRY, item.label, byte_count, attempt,
                    tx_time_s, s.elevation_deg, parsed=parsed,
                    error="CRC failed repeatedly — discarded"
                )
            return TransmitResult(
                TxStatus.CRC_FAIL, item.label, byte_count, attempt,
                tx_time_s, s.elevation_deg, parsed=parsed,
                error=f"CRC mismatch #{attempt} — retransmitting"
            )

        # Success
        return TransmitResult(
            TxStatus.SUCCESS, item.label, byte_count, attempt,
            tx_time_s, s.elevation_deg, parsed=parsed
        )

    def _transmit_raw_chunk(self, s: RFLinkState) -> TransmitResult:
        """
        Simulate sending a chunk of raw imagery.
        Each tick we can send at most BYTES_PER_TICK bytes.
        Progress is tracked as percentage of RAW_IMAGE_BYTES.
        """
        chunk = min(BYTES_PER_TICK, RAW_IMAGE_BYTES - self._raw_sent_bytes)
        # Apply packet loss to raw chunks too
        if self._rng.random() < PACKET_LOSS_RATE:
            chunk = 0   # lost chunk — no retransmit for raw (too slow)

        self._raw_sent_bytes += chunk
        tx_time_s = (chunk * 8) / DATA_RATE_BPS if chunk else 0

        return TransmitResult(
            TxStatus.SUCCESS if chunk > 0 else TxStatus.DROPPED,
            RAW_IMAGE_LABEL,
            chunk,
            1,
            tx_time_s,
            s.elevation_deg,
        )

    def _raw_progress_pct(self) -> float:
        return round(min(100.0, self._raw_sent_bytes / RAW_IMAGE_BYTES * 100), 4)

    def snapshot(self, s: RFLinkState) -> dict:
        d = {
            "pass_open":         s.pass_open,
            "elevation_deg":     s.elevation_deg,
            "window_elapsed_s":  s.window_elapsed_s,
            "bytes_sent_tick":   s.bytes_sent_tick,
            "bytes_budget_tick": s.bytes_budget_tick,
            "queue_depth":       s.queue_depth,
            "raw_progress_pct":  s.raw_progress_pct,
            "total_bytes_sent":  s.total_bytes_sent,
            "pass_number":       s.pass_number,
        }
        if s.last_tx:
            d["last_tx_status"]  = s.last_tx.status.value
            d["last_tx_bytes"]   = s.last_tx.byte_count
            d["last_tx_time_s"]  = round(s.last_tx.tx_time_s, 4)
            d["last_tx_label"]   = s.last_tx.label
            d["last_tx_error"]   = s.last_tx.error
        return d
