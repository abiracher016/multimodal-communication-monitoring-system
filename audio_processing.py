"""
Audio Processing Module
- Whisper transcription (file & live chunks)
- Speaker diarization (pyannote with fallback)
- Dynamic role mapping
- Live microphone capture
"""
import os
import json
import tempfile
import traceback
import numpy as np
from datetime import datetime

try:
    import whisper
except ImportError:
    whisper = None

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError:
    sd = None
    wav_write = None

from backend.config import (
    WHISPER_MODEL, SAMPLE_RATE, CHUNK_DURATION, HF_TOKEN
)


# ─── Whisper Model Singleton ─────────────────────────────────────────────────
_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        if whisper is None:
            raise ImportError("openai-whisper is not installed. Run: pip install openai-whisper")
        print(f"[Audio] Loading Whisper model '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("[Audio] Whisper model loaded.")
    return _whisper_model


# ─── Transcription ────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file using Whisper.
    Returns list of segment dicts: {start, end, text}
    """
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": text,
            })
    return segments


def transcribe_audio_chunk(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a numpy audio array (for live streaming)."""
    # Save to temp file, transcribe, delete
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if wav_write is None:
            raise ImportError("scipy not installed")
        wav_write(tmp_path, sample_rate, audio_array)
        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        return result.get("text", "").strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── Speaker Diarization ─────────────────────────────────────────────────────
def diarize_audio(audio_path: str) -> list[dict]:
    """
    Run speaker diarization. Uses pyannote if HF_TOKEN is set,
    otherwise falls back to energy-based segmentation.
    """
    if HF_TOKEN:
        try:
            return _diarize_pyannote(audio_path)
        except Exception as e:
            print(f"[Audio] pyannote diarization failed: {e}")
            print("[Audio] Falling back to energy-based segmentation...")

    return _diarize_energy_fallback(audio_path)


def _diarize_pyannote(audio_path: str) -> list[dict]:
    """Diarize using pyannote/speaker-diarization."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })
    return segments


def _diarize_energy_fallback(audio_path: str) -> list[dict]:
    """
    Simple energy-based speaker segmentation fallback.
    Assigns alternating speakers based on silence gaps.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except ImportError:
        # If librosa not available, create simple 2-speaker alternation
        return _simple_alternation_fallback(audio_path)

    # Split by silence intervals
    intervals = librosa.effects.split(y, top_db=30)
    segments = []
    current_speaker = "SPEAKER_00"
    last_end = 0.0

    for start_sample, end_sample in intervals:
        start_time = round(start_sample / sr, 2)
        end_time = round(end_sample / sr, 2)

        # Switch speaker if gap > 1.5s
        if start_time - last_end > 1.5:
            current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"

        segments.append({
            "start": start_time,
            "end": end_time,
            "speaker": current_speaker,
        })
        last_end = end_time

    return segments


def _simple_alternation_fallback(audio_path: str) -> list[dict]:
    """Absolute fallback — alternates speakers every 15 seconds."""
    try:
        import wave
        with wave.open(audio_path, 'r') as wav:
            duration = wav.getnframes() / wav.getframerate()
    except Exception:
        duration = 300  # default 5 minutes

    segments = []
    current = 0.0
    speaker_idx = 0
    while current < duration:
        end = min(current + 15.0, duration)
        segments.append({
            "start": round(current, 2),
            "end": round(end, 2),
            "speaker": f"SPEAKER_0{speaker_idx}",
        })
        current = end
        speaker_idx = 1 - speaker_idx
    return segments


# ─── Speaker Matching ─────────────────────────────────────────────────────────
def match_speaker(seg_start: float, seg_end: float, diarization_segments: list[dict]) -> str:
    """Match a transcription segment to a diarized speaker."""
    mid = (seg_start + seg_end) / 2
    for sp in diarization_segments:
        if sp["start"] <= mid <= sp["end"]:
            return sp["speaker"]
    # Fallback: find closest
    if diarization_segments:
        closest = min(diarization_segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - mid))
        return closest["speaker"]
    return "UNKNOWN"


def map_roles(segments: list[dict]) -> dict:
    """
    Map raw speaker IDs to roles.
    Most-talking speaker → Candidate, second → Interviewer.
    """
    from collections import Counter
    speaker_counts = Counter(seg.get("speaker_raw", seg.get("speaker", "")) for seg in segments)
    sorted_speakers = speaker_counts.most_common()

    role_map = {}
    if len(sorted_speakers) >= 1:
        role_map[sorted_speakers[0][0]] = "Candidate"
    if len(sorted_speakers) >= 2:
        role_map[sorted_speakers[1][0]] = "Interviewer"
    for i, (spk, _) in enumerate(sorted_speakers[2:], start=3):
        role_map[spk] = f"Speaker {i}"

    return role_map


# ─── Live Capture ─────────────────────────────────────────────────────────────
def record_audio_chunk(duration: float = CHUNK_DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone."""
    if sd is None:
        raise ImportError("sounddevice not installed. Run: pip install sounddevice")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


# ─── Full Pipeline ────────────────────────────────────────────────────────────
def process_audio_file(audio_path: str) -> list[dict]:
    """
    Full audio processing pipeline:
    1. Transcribe with Whisper
    2. Diarize speakers
    3. Match speakers to segments
    4. Map roles
    Returns enriched segments list.
    """
    print(f"[Audio] Processing: {audio_path}")

    # Step 1: Transcribe
    transcript_segments = transcribe_audio(audio_path)
    print(f"[Audio] Transcribed {len(transcript_segments)} segments")

    # Step 2: Diarize
    diarization_segments = diarize_audio(audio_path)
    print(f"[Audio] Diarized {len(diarization_segments)} speaker segments")

    # Step 3: Match speakers
    for seg in transcript_segments:
        seg["speaker_raw"] = match_speaker(seg["start"], seg["end"], diarization_segments)

    # Step 4: Map roles
    role_map = map_roles(transcript_segments)
    print(f"[Audio] Role map: {role_map}")

    for seg in transcript_segments:
        seg["speaker"] = role_map.get(seg["speaker_raw"], seg["speaker_raw"])
        seg["timestamp"] = datetime.now().isoformat()

    return transcript_segments
