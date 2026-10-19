"""Turns a call recording into a timestamped, speaker-labeled transcript."""
import assemblyai as aai

from analyzer import get_secret


def fmt_time(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def to_seconds(stamp: str) -> int:
    """'1:05' -> 65. Returns 0 if the stamp is empty or malformed."""
    try:
        minutes, seconds = stamp.strip().strip("[]").split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return 0


def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to AssemblyAI with speaker labels and return text lines like
    '[0:12] Speaker A: Hi, this is William with Northstar Ads.'"""
    aai.settings.api_key = get_secret("ASSEMBLYAI_API_KEY")
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(audio_bytes, config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    lines = [f"[{fmt_time(u.start)}] Speaker {u.speaker}: {u.text}"
             for u in (transcript.utterances or [])]
    if not lines:
        raise RuntimeError("No speech was detected in this recording.")
    return "\n".join(lines)
