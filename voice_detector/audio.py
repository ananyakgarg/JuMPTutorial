"""Audio loading and preprocessing shared by training and inference."""

import io
import subprocess

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 2


def load_audio(source, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a file path or encoded audio bytes as mono float32 at `sr` Hz."""
    try:
        data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        wav, file_sr = sf.read(data, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except sf.LibsndfileError:
        # libsndfile can't read m4a/aac and some mp3s; ffmpeg can.
        return _decode_with_ffmpeg(source, sr)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _decode_with_ffmpeg(source, sr: int) -> np.ndarray:
    is_bytes = isinstance(source, (bytes, bytearray))
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0" if is_bytes else str(source),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    out = subprocess.run(cmd, input=source if is_bytes else None, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def preprocess(wav: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Trim leading/trailing silence and peak-normalise.

    Real and synthetic corpora differ systematically in how much silence surrounds
    each clip and how loud they are. Removing both stops the model from learning
    those shortcuts instead of properties of the voice itself.
    """
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    if len(trimmed) >= MIN_SAMPLES:
        wav = trimmed
    if len(wav) < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)))
    peak = np.abs(wav).max()
    return wav / peak * 0.95 if peak > 0 else wav


def windows(wav: np.ndarray, seconds: float, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """Split into consecutive windows of `seconds`.

    A trailing piece shorter than half a window is dropped, unless it is the only one.
    """
    size = int(seconds * sr)
    chunks = [wav[i:i + size] for i in range(0, len(wav), size)]
    if len(chunks) > 1 and len(chunks[-1]) < size // 2:
        chunks.pop()
    return chunks
