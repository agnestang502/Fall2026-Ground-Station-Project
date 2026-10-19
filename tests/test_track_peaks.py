"""
Checks for track_peaks.py using fake SatNOGS-style recordings whose residual we know.
Run with:  python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import track_peaks as tp  # noqa: E402

FS = 48000
DURATION = 120  # seconds


def satnogs_audio(tmp_path, name, residual, bpsk=True, snr_amp=0.05, extra=None, seed=0):
    """
    Build audio the way SatNOGS does: a signal at `residual(t)` Hz from the expected
    frequency, shifted up by 12 kHz, real part, plus noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(DURATION * FS) / FS
    phase = 2 * np.pi * np.cumsum(residual(t)) / FS
    bb = np.exp(1j * phase)
    if bpsk:                                       # 1200 baud random bits: +1 / -1
        bits = rng.choice([-1.0, 1.0], size=int(DURATION * 1200) + 1)
        bb = bb * bits[(t * 1200).astype(int)]
    x = snr_amp * np.real(bb * np.exp(2j * np.pi * tp.IF_HZ * t)) + rng.normal(0, 0.1, t.size)
    if extra is not None:
        x = x + extra(t)
    path = tmp_path / f"{name}.wav"
    sf.write(path, (0.5 * x).astype(np.float32), FS)
    return path


def tracked(path, mode="BPSK"):
    r = tp.track_pass(path, {"transmitter_mode": mode})
    t, f, _ = r["all"][r["good"]].T
    return t, f, r


def test_constant_bpsk_offset_is_measured(tmp_path):
    t, f, r = tracked(satnogs_audio(tmp_path, "const", lambda t: -350.0 + 0 * t))
    assert r["good"].mean() > 0.9
    assert abs(np.median(f) + 350.0) < 0.5
    assert np.std(f) < 1.0


def test_s_shaped_residual_is_followed(tmp_path):
    # A residual that swings by 200 Hz around closest approach at t = 60 s.
    truth = lambda t: 150.0 + 100.0 * np.tanh((t - 60) / 15)   # noqa: E731
    t, f, r = tracked(satnogs_audio(tmp_path, "s", truth))
    assert r["good"].mean() > 0.9
    assert np.max(np.abs(f - truth(t))) < 3.0      # within a few Hz everywhere
    assert np.median(np.abs(f - truth(t))) < 1.0


def test_plain_carrier_mode(tmp_path):
    t, f, _ = tracked(satnogs_audio(tmp_path, "cw", lambda t: 420.0 + 0.5 * t, bpsk=False), mode="CW")
    assert np.median(np.abs(f - (420.0 + 0.5 * t))) < 0.5


def test_noise_only_gives_no_track(tmp_path):
    t, f, r = tracked(satnogs_audio(tmp_path, "noise", lambda t: 0 * t, snr_amp=0.0))
    assert r["summary"]["points"] < 10


def test_other_signal_is_not_followed(tmp_path):
    # FUNcube-like signal only in the first half; a strong steady tone 1.5 kHz away
    # throughout. The tracker must not jump to the tone when the satellite stops.
    def other(t):
        return 0.05 * np.sin(2 * np.pi * (tp.IF_HZ + 1500) * t)

    rng = np.random.default_rng(3)
    path = satnogs_audio(tmp_path, "cross", lambda t: 200.0 + 0 * t, extra=other)
    x, fs = sf.read(path)
    x[len(x) // 2:] = 0.5 * rng.normal(0, 0.1, len(x) - len(x) // 2) + \
        0.5 * other(np.arange(len(x) // 2, len(x)) / fs)
    sf.write(path, x.astype(np.float32), fs)
    t, f, _ = tracked(path)
    assert np.all(np.abs(f - 200.0) < 5.0)
    assert t.max() < DURATION / 2 + 2
