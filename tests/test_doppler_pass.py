"""Offline checks for doppler_pass.py, using the ISS example TLE from skyfield's docs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doppler_pass as dp  # noqa: E402
import predict_passes as pp  # noqa: E402
from test_find_satellite import iss  # noqa: E402

STATION = pp.make_station(40.0, -75.0, 100)     # roughly Philadelphia
F0 = 437_625_000


def highest_pass(tmp_path):
    sat = iss(tmp_path)
    passes = pp.find_passes(sat, STATION, sat.epoch, sat.epoch + 2)
    return sat, passes, max(passes, key=lambda p: p["peak_deg"])


def test_dot_product_by_hand():
    origin = np.zeros((3, 2))
    r_sat = np.array([[1e6, 0.0], [0.0, 1e6], [0.0, 0.0]])          # on the x axis, then the y axis
    v_sat = np.array([[1000.0, 1000.0], [0.0, 0.0], [0.0, 0.0]])    # both moving along +x
    rng, rate = dp.range_rate_dot(r_sat, v_sat, origin, origin)
    assert np.allclose(rng, 1e6)
    assert np.allclose(rate, [1000.0, 0.0])      # straight away, then sideways


def test_doppler_sign():
    f_r, shift = dp.doppler_hz(F0, np.array([-7500.0, 0.0, 7500.0]))
    assert shift[0] > 0 and shift[1] == 0 and shift[2] < 0   # approaching = higher pitch
    assert abs(shift[0] - dp.reference_shift_hz(F0)) < 1e-6
    assert np.allclose(f_r - F0, shift)


def test_three_range_rates_agree(tmp_path):
    sat, _, best = highest_pass(tmp_path)
    a = dp.analyse_pass(sat, STATION, best, F0)
    dot = a["rates"]["dot product"]
    assert np.max(np.abs(a["rates"]["skyfield"] - dot)) < 1e-3
    assert np.max(np.abs(a["rates"]["finite difference"] - dot)) < dp.AGREE_M_S
    assert np.all(np.diff(dot) > 0)              # range-rate rises steadily through the pass


def test_no_sawtooth_from_time_rounding(tmp_path):
    # With the date split into whole day + fraction, the finite-difference error is
    # smooth: neighbouring samples never jump by more than a few mm/s.
    sat, _, best = highest_pass(tmp_path)
    a = dp.analyse_pass(sat, STATION, best, F0)
    err = a["rates"]["finite difference"] - a["rates"]["dot product"]
    assert np.max(np.abs(np.diff(err[2:-2]))) < 0.02


def test_highest_pass_passes_checkpoint(tmp_path):
    sat, _, best = highest_pass(tmp_path)
    a = dp.analyse_pass(sat, STATION, best, F0)
    problems, found = dp.checkpoint(a["seconds"], a["range_m"], a["rates"], a["shift"], F0)
    assert problems == []
    assert abs(found["zero_cross_s"] - found["min_range_s"]) < 0.5
    assert 0.8 < found["peak_ratio"] < 1.05


def test_checkpoint_flags_flipped_sign(tmp_path):
    sat, _, best = highest_pass(tmp_path)
    a = dp.analyse_pass(sat, STATION, best, F0)
    problems, _ = dp.checkpoint(a["seconds"], a["range_m"], a["rates"], -a["shift"], F0)
    assert any("start positive" in m for m in problems)


def test_min_range_and_zero_crossing_helpers():
    s = np.arange(0.0, 11.0)
    assert abs(dp.min_range_s(s, (s - 4.3) ** 2 + 5) - 4.3) < 1e-9
    assert abs(dp.zero_crossing_s(s, 4.3 - s) - 4.3) < 1e-9
    assert dp.zero_crossing_s(s, s + 1) is None


def test_plots(tmp_path):
    sat, passes, best = highest_pass(tmp_path)
    a = dp.analyse_pass(sat, STATION, best, F0)
    _, found = dp.checkpoint(a["seconds"], a["range_m"], a["rates"], a["shift"], F0)
    assert dp.plot_range_rate(a["seconds"], a["rates"], best, "t", tmp_path / "rr.png").stat().st_size > 10_000
    assert dp.plot_doppler(a["seconds"], a["shift"], a["range_m"], found, F0, best, "t",
                           tmp_path / "d.png").stat().st_size > 10_000
    html = dp.plot_3d(sat, STATION, best, "t", tmp_path / "p.html")
    assert "plotly" in html.read_text()[:5000].lower()
    low = min(passes, key=lambda p: p["peak_deg"])
    out, rows = dp.plot_high_low(sat, STATION, best, low, F0, "t", tmp_path / "hl.png")
    assert out.stat().st_size > 10_000
    (_, _, close_hi, steep_hi, _), (_, _, close_lo, steep_lo, _) = rows
    assert close_hi < close_lo and steep_hi < steep_lo < 0   # the high pass flips faster
