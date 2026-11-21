"""
Module 7 — Ground Dispatch: Vernacular Alert Generation
=========================================================
Receives a validated ParsedPacket from the RF link and synthesises a
spoken tactical alert in the local language.

Two TTS backends, selected automatically:

  Backend A — Sarvam Bulbul API (preferred)
    Uses POST https://api.sarvam.ai/text-to-speech
    Requires SARVAM_API_KEY in environment or passed at runtime.
    Produces real Bulbul-quality audio in Kannada / Hindi / English.
    Label in UI: "Sarvam Bulbul (cloud-connected)"

  Backend B — Browser / System TTS (fallback)
    Emits a structured JSON payload that the browser's Web Speech API
    will synthesise. The Python server does NOT call Web Speech —
    it only produces the text and language tag. The browser picks up
    the JSON via WebSocket and runs speechSynthesis.speak() there.
    Label in UI: "Browser TTS (offline fallback — voice varies by device)"

    Important: we do NOT claim "Bulbul via Web Speech API."
    If a local voice is available the browser will use it; if not,
    the browser may use a remote voice. The UI checks voice.localService
    and labels accordingly.

Alert text is generated from the ParsedPacket fields — no hardcoded strings.
Supported languages: Kannada (kn-IN), Hindi (hi-IN), English (en-IN).

Output (DispatchResult):
  backend_used   — "bulbul" | "browser_tts"
  language       — BCP-47 tag
  alert_text     — the full spoken text (native script)
  audio_b64      — base64-encoded WAV/MP3 (Bulbul only, else None)
  packet_summary — structured dict of the event for the map pin
  success        — bool
  error          — error string if failed
"""

import os
import json
import base64
import http.client
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

from modules.packet import ParsedPacket, FLAG_RADIATION_SEEN, FLAG_ECLIPSE

# ── Language config ────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "kn-IN": "Kannada",
    "hi-IN": "Hindi",
    "en-IN": "English",
}

DEFAULT_LANGUAGE = "en-IN"

# Sarvam Bulbul API
BULBUL_ENDPOINT  = "api.sarvam.ai"
BULBUL_PATH      = "/text-to-speech"
BULBUL_MODEL     = "bulbul:v1"
BULBUL_SPEAKER   = "anushka"   # Female voice, clear diction


# ── Alert text templates (per language) ───────────────────────────────────────
# Templates use {placeholders} filled from ParsedPacket fields.
# Kept short and tactical — a ranger hears this over VHF radio.

def _build_alert_text(parsed: ParsedPacket, lang: str) -> str:
    """Generate a spoken tactical alert from packet fields."""
    ev      = parsed.event_class
    conf    = int(parsed.confidence * 100)
    lat     = parsed.latitude
    lon     = parsed.longitude
    sev_map = {0: "low", 85: "high", 70: "medium"}
    sev     = "high" if parsed.severity >= 217 else "medium" if parsed.severity >= 178 else "low"

    # Event names and action phrases
    if lang == "kn-IN":
        events = {
            "WILDFIRE":      "ಕಾಡ್ಗಿಚ್ಚು",
            "PIPELINE_LEAK": "ಪೈಪ್‌ಲೈನ್ ಸೋರಿಕೆ",
            "CROP_STRESS":   "ಬೆಳೆ ಒತ್ತಡ",
            "FLOOD":         "ಪ್ರವಾಹ",
        }
        actions = {
            "WILDFIRE":      "ತಕ್ಷಣ ಪ್ರದೇಶ ಖಾಲಿ ಮಾಡಿ",
            "PIPELINE_LEAK": "ಪೈಪ್‌ಲೈನ್ ಮುಚ್ಚಿ ತನಿಖೆ ಮಾಡಿ",
            "CROP_STRESS":   "ಸಮೀಕ್ಷಾ ತಂಡವನ್ನು ಕಳುಹಿಸಿ",
            "FLOOD":         "ರಸ್ತೆಗಳನ್ನು ಮುಚ್ಚಿ ಜನರನ್ನು ಎಚ್ಚರಿಸಿ",
        }
        ev_name = events.get(ev, ev)
        action  = actions.get(ev, "ಪರಿಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ")
        return (
            f"ಎಚ್ಚರಿಕೆ. ಉಪಗ್ರಹ ಪತ್ತೆ: {ev_name}. "
            f"ಸ್ಥಳ: ಅಕ್ಷಾಂಶ {lat:.2f}, ರೇಖಾಂಶ {lon:.2f}. "
            f"ವಿಶ್ವಾಸ ಮಟ್ಟ {conf} ಶೇಕಡ. "
            f"ಕ್ರಮ: {action}."
        )

    elif lang == "hi-IN":
        events = {
            "WILDFIRE":      "जंगल की आग",
            "PIPELINE_LEAK": "पाइपलाइन रिसाव",
            "CROP_STRESS":   "फसल तनाव",
            "FLOOD":         "बाढ़",
        }
        actions = {
            "WILDFIRE":      "तुरंत क्षेत्र खाली करें",
            "PIPELINE_LEAK": "पाइपलाइन बंद कर जांच करें",
            "CROP_STRESS":   "सर्वेक्षण दल भेजें",
            "FLOOD":         "सड़कें बंद करें और लोगों को सचेत करें",
        }
        ev_name = events.get(ev, ev)
        action  = actions.get(ev, "स्थिति की जांच करें")
        return (
            f"चेतावनी। उपग्रह से पता चला: {ev_name}। "
            f"स्थान: अक्षांश {lat:.2f}, देशांतर {lon:.2f}। "
            f"विश्वास स्तर {conf} प्रतिशत। "
            f"कार्रवाई: {action}।"
        )

    else:  # en-IN default
        events = {
            "WILDFIRE":      "wildfire",
            "PIPELINE_LEAK": "pipeline leak",
            "CROP_STRESS":   "crop stress",
            "FLOOD":         "flood",
        }
        actions = {
            "WILDFIRE":      "evacuate the zone immediately",
            "PIPELINE_LEAK": "isolate pipeline segment and inspect",
            "CROP_STRESS":   "dispatch ground survey team",
            "FLOOD":         "close roads and alert district",
        }
        ev_name = events.get(ev, ev.lower())
        action  = actions.get(ev, "assess situation")
        return (
            f"Alert. Satellite detection: {ev_name}. "
            f"Location: latitude {lat:.2f}, longitude {lon:.2f}. "
            f"Confidence {conf} percent. "
            f"Action: {action}."
        )


# ── Bulbul API call ────────────────────────────────────────────────────────────

def _call_bulbul(text: str, lang: str, api_key: str) -> tuple[bool, Optional[bytes], str]:
    """
    POST to Sarvam Bulbul TTS endpoint.
    Returns (success, audio_bytes_or_None, error_string).
    Audio is base64-decoded from the API response.
    """
    payload = json.dumps({
        "inputs":        [text],
        "target_language_code": lang,
        "speaker":       BULBUL_SPEAKER,
        "model":         BULBUL_MODEL,
        "enable_preprocessing": True,
    }).encode("utf-8")

    headers = {
        "Content-Type":  "application/json",
        "api-subscription-key": api_key,
    }

    try:
        conn = http.client.HTTPSConnection(BULBUL_ENDPOINT, timeout=15)
        conn.request("POST", BULBUL_PATH, body=payload, headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        if resp.status != 200:
            return False, None, f"Bulbul API {resp.status}: {body[:200].decode(errors='replace')}"

        data = json.loads(body)
        # Bulbul returns: {"audios": ["<base64-encoded-wav>"]}
        audios = data.get("audios", [])
        if not audios:
            return False, None, "Bulbul returned empty audios list"

        audio_bytes = base64.b64decode(audios[0])
        return True, audio_bytes, ""

    except Exception as e:
        return False, None, f"Bulbul API error: {e}"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    backend_used:   str            # "bulbul" | "browser_tts"
    language:       str            # BCP-47 tag
    language_name:  str            # "Kannada" etc.
    alert_text:     str            # spoken text
    audio_b64:      Optional[str]  # base64 audio (Bulbul only)
    packet_summary: dict           # structured event data for map/UI
    success:        bool
    error:          str = ""


# ── Main Engine ────────────────────────────────────────────────────────────────

class GroundDispatch:
    """
    Call .dispatch(parsed_packet, language) to generate a vernacular alert.

    Key selection:
      1. Checks SARVAM_API_KEY env var.
      2. If key present → Bulbul API (labeled "cloud-connected").
      3. If key absent  → browser_tts JSON payload (labeled "offline fallback").

    The api_key parameter overrides the env var (for UI-supplied keys).
    """

    def __init__(self, api_key: Optional[str] = None):
        env_key = os.environ.get("SARVAM_API_KEY", "").strip()
        self._api_key = api_key or env_key or None

    def set_api_key(self, key: str):
        self._api_key = key.strip() or None

    @property
    def backend(self) -> str:
        return "bulbul" if self._api_key else "browser_tts"

    def dispatch(self,
                 parsed: ParsedPacket,
                 language: str = DEFAULT_LANGUAGE) -> DispatchResult:
        """
        Generate and (optionally) synthesise a vernacular alert.

        Parameters
        ----------
        parsed   : a valid ParsedPacket from rf_link / packet.deserialize
        language : BCP-47 language code (kn-IN / hi-IN / en-IN)
        """
        if language not in SUPPORTED_LANGUAGES:
            language = DEFAULT_LANGUAGE

        lang_name  = SUPPORTED_LANGUAGES[language]
        alert_text = _build_alert_text(parsed, language)

        # Build structured packet summary for the map pin
        packet_summary = {
            "event_class":  parsed.event_class,
            "severity":     parsed.severity,
            "confidence":   parsed.confidence,
            "latitude":     parsed.latitude,
            "longitude":    parsed.longitude,
            "bbox_lat":     parsed.bbox_lat,
            "bbox_lon":     parsed.bbox_lon,
            "flags": {
                "radiation_seen":      bool(parsed.flags & FLAG_RADIATION_SEEN),
                "inference_throttled": bool(parsed.flags & 0b010),
                "eclipse":             bool(parsed.flags & 0b100),
            },
            "summary":      parsed.summary,
            "alert_text":   alert_text,
            "language":     language,
            "language_name":lang_name,
        }

        # ── Backend selection ──────────────────────────────────────────────
        if self._api_key:
            return self._dispatch_bulbul(
                alert_text, language, lang_name, packet_summary
            )
        else:
            return self._dispatch_browser_tts(
                alert_text, language, lang_name, packet_summary
            )

    # ── Backend implementations ────────────────────────────────────────────────

    def _dispatch_bulbul(self, text, lang, lang_name, summary) -> DispatchResult:
        success, audio_bytes, error = _call_bulbul(text, lang, self._api_key)

        if success:
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        else:
            audio_b64 = None

        return DispatchResult(
            backend_used  = "bulbul",
            language      = lang,
            language_name = lang_name,
            alert_text    = text,
            audio_b64     = audio_b64,
            packet_summary= summary,
            success       = success,
            error         = error,
        )

    def _dispatch_browser_tts(self, text, lang, lang_name, summary) -> DispatchResult:
        """
        Produce a JSON payload the browser WebSocket client will pick up
        and pass to speechSynthesis.speak(). No audio is generated server-side.
        The browser will check voice.localService before claiming offline capability.
        """
        return DispatchResult(
            backend_used  = "browser_tts",
            language      = lang,
            language_name = lang_name,
            alert_text    = text,
            audio_b64     = None,       # browser synthesises locally
            packet_summary= summary,
            success       = True,       # text generation always succeeds
            error         = "",
        )

    def to_ws_payload(self, result: DispatchResult) -> dict:
        """
        Serialise a DispatchResult to a dict suitable for JSON WebSocket push.
        The browser client reads this and either plays audio_b64 or calls
        speechSynthesis.speak(alert_text) depending on backend_used.
        """
        return {
            "type":          "ground_alert",
            "backend":       result.backend_used,
            "language":      result.language,
            "language_name": result.language_name,
            "alert_text":    result.alert_text,
            "audio_b64":     result.audio_b64,
            "success":       result.success,
            "error":         result.error,
            "packet":        result.packet_summary,
        }
