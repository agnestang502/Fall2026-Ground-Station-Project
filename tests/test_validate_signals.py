"""
Checks for validate_signals.py using synthetic recordings whose answer we know.
Run with:  python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_signals as vs  # noqa: E402

FS = 48000
DURATION = 180  # seconds


def make_recording(tmp_path, name, tone=None, dropouts=False, seed=0):
    """
    White noise (like an empty pass) plus, optionally, a tone.
    tone = (amplitude, start_hz, drift_hz_per_s, on_from_s, on_until_s)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(DURATION * FS) / FS
    x = rng.normal(0, 0.1, t.size)
    if tone:
        amp, f0, drift, on, off = tone
        phase = 2 * np.pi * (f0 * t + 0.5 * drift * t ** 2)
        gate = (t >= on) & (t < off)
        x += gate * amp * np.sin(phase)
    if dropouts:
        for start in range(10, DURATION, 30):          # 6 s of silence every 30 s
            x[start * FS:(start + 6) * FS] = 0
    path = tmp_path / f"{name}.wav"   # WAV: libsndfile can crash writing long OGG files
    sf.write(path, x.astype(np.float32) * 0.5, FS)
    return path


def verdict(path, metadata=None):
    return vs.decide(vs.analyse_audio(path), metadata)[0]


@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_is_rejected(tmp_path, seed):
    assert verdict(make_recording(tmp_path, f"noise{seed}", seed=seed)) == "NO_SIGNAL"


def test_drifting_satellite_like_tone_is_accepted(tmp_path):
    # Weak tone (7% of the noise amplitude), drifting like residual Doppler,
    # present for the middle 2/3 of the pass.
    path = make_recording(tmp_path, "sat", tone=(0.007, 1500, -0.3, 30, 150))
    assert verdict(path) == "SIGNAL"


def test_too_weak_tone_is_rejected(tmp_path):
    path = make_recording(tmp_path, "weak", tone=(0.003, 1500, -0.3, 30, 150))
    assert verdict(path) == "NO_SIGNAL"


def test_too_short_signal_is_rejected(tmp_path):
    path = make_recording(tmp_path, "short", tone=(0.01, 1500, -0.3, 60, 70))
    assert verdict(path) == "NO_SIGNAL"


def test_constant_tone_is_not_trusted(tmp_path):
    path = make_recording(tmp_path, "spur", tone=(0.01, 2000, 0.0, 0, DURATION))
    assert verdict(path) == "UNCERTAIN"


def test_dropouts_make_pass_uncertain(tmp_path):
    path = make_recording(tmp_path, "gaps", tone=(0.01, 1500, -0.3, 0, DURATION), dropouts=True)
    assert verdict(path) == "UNCERTAIN"


def test_other_satellite_crossing_is_not_trusted(tmp_path):
    # An uncorrected satellite sweeping across the band (S-shaped Doppler curve).
    rng = np.random.default_rng(1)
    t = np.arange(DURATION * FS) / FS
    freq = 6000 - 4000 * np.tanh((t - 90) / 20)
    x = rng.normal(0, 0.1, t.size) + 0.01 * np.sin(2 * np.pi * np.cumsum(freq) / FS)
    path = tmp_path / "crossing.wav"
    sf.write(path, x.astype(np.float32) * 0.5, FS)
    assert verdict(path) != "SIGNAL"


def test_satnogs_without_signal_vote_is_respected(tmp_path):
    path = make_recording(tmp_path, "vetted", tone=(0.01, 1500, -0.3, 30, 150))
    assert verdict(path, {"waterfall_status": "without-signal"}) == "UNCERTAIN"


def make_wide_signal(tmp_path, name, amp, seed=0):
    """Noise plus a steady 1 kHz-wide band at 11.5 kHz, like FUNcube-1's BPSK telemetry."""
    from scipy import signal as sps
    rng = np.random.default_rng(seed)
    t = np.arange(DURATION * FS) / FS
    sos = sps.butter(6, [11000, 12000], btype="bandpass", fs=FS, output="sos")
    band = sps.sosfilt(sos, rng.normal(0, 1, t.size))
    gate = (t >= 30) & (t < 150)
    x = rng.normal(0, 0.1, t.size) + amp * gate * band / band.std()
    path = tmp_path / f"{name}.wav"
    sf.write(path, x.astype(np.float32) * 0.5, FS)
    return path


def test_wide_bpsk_like_band_is_accepted(tmp_path):
    # Band power ~2x the noise in the same frequencies (about +5 dB).
    assert verdict(make_wide_signal(tmp_path, "bpsk", amp=0.1 * np.sqrt(2 / 48 * 2))) == "SIGNAL"


def test_too_faint_wide_band_is_not_accepted(tmp_path):
    assert verdict(make_wide_signal(tmp_path, "faint", amp=0.1 * np.sqrt(0.1 / 48 * 2))) != "SIGNAL"


def test_cnn_screen_rejects_low_scores(tmp_path):
    path = make_recording(tmp_path, "sat", tone=(0.007, 1500, -0.3, 30, 150))
    measure = vs.analyse_audio(path)
    assert vs.decide(measure, None, cnn_prob=vs.CNN_THRESHOLD - 0.01)[0] == "NO_SIGNAL"
    assert vs.decide(measure, None, cnn_prob=vs.CNN_THRESHOLD + 0.01)[0] == "SIGNAL"


def test_high_cnn_score_alone_is_not_enough(tmp_path):
    path = make_recording(tmp_path, "noise", seed=7)
    assert vs.decide(vs.analyse_audio(path), None, cnn_prob=0.99)[0] == "NO_SIGNAL"


def test_metadata_check():
    assert vs.metadata_status(None)
    full = {"station_lat": 21.3, "station_lng": -157.8, "station_alt": 10,
            "tle1": "1 39444U ...", "tle2": "2 39444 ..."}
    assert vs.metadata_status(full) == []
    assert vs.metadata_status({**full, "tle1": ""}) == ["tle1"]
