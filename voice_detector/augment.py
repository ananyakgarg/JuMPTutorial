"""Random degradations for training audio.

The human clips in MLAAD are clean audiobook recordings, so a detector trained on
them as-is can learn "clean studio audio = human". Real-world audio (phone calls,
social media, compressed video) is noisy and lossy whether the voice is real or
fake, so training clips are passed through the same kinds of damage.
"""

import subprocess

import librosa
import numpy as np
from scipy.signal import fftconvolve

from .audio import SAMPLE_RATE, load_audio


def codec(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round-trip through a lossy codec at a random bitrate."""
    fmt, encoder, bitrate = [("mp3", "libmp3lame", rng.choice([32, 64, 96, 128])),
                             ("ogg", "libopus", rng.choice([12, 24, 32, 48]))][rng.integers(2)]
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
           "-c:a", encoder, "-b:a", f"{bitrate}k", "-f", fmt, "pipe:1"]
    encoded = subprocess.run(cmd, input=wav.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return load_audio(encoded)[:len(wav)]


def noise(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add white or brown-ish noise at 5-30 dB SNR."""
    n = rng.standard_normal(len(wav))
    if rng.random() < 0.5:
        n = np.cumsum(n)
        n -= np.convolve(n, np.ones(1000) / 1000, mode="same")  # remove drift
    snr_db = rng.uniform(5, 30)
    scale = np.sqrt(np.mean(wav ** 2) / (np.mean(n ** 2) + 1e-12) / 10 ** (snr_db / 10))
    return wav + scale * n


def telephone(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Band-limit to 8 kHz sampling (4 kHz bandwidth), like a phone line."""
    narrow = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=8000)
    return librosa.resample(narrow, orig_sr=8000, target_sr=SAMPLE_RATE)[:len(wav)]


def reverb(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic room impulse response (RT60 0.2-0.8 s)."""
    rt60 = rng.uniform(0.2, 0.8)
    t = np.arange(int(rt60 * SAMPLE_RATE)) / SAMPLE_RATE
    impulse = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    impulse[0] = 1.0
    return fftconvolve(wav, impulse)[:len(wav)]


DEGRADATIONS = [codec, noise, telephone, reverb]


def degrade(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one or two random degradations, then re-normalise the peak level."""
    for i in rng.choice(len(DEGRADATIONS), size=rng.integers(1, 3), replace=False):
        wav = DEGRADATIONS[i](wav, rng)
    peak = np.abs(wav).max()
    return (wav / peak * 0.95 if peak > 0 else wav).astype(np.float32)
