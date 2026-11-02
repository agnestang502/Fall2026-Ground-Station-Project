"""Offline checks for predict_passes.py, using the ISS example TLE from skyfield's docs."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict_passes as pp  # noqa: E402
from test_find_satellite import iss  # noqa: E402

STATION = pp.make_station(40.0, -75.0, 100)     # roughly Philadelphia


def passes_over_two_days(tmp_path):
    sat = iss(tmp_path)
    return sat, pp.find_passes(sat, STATION, sat.epoch, sat.epoch + 2)


def test_passes_are_ordered_and_complete(tmp_path):
    sat, passes = passes_over_two_days(tmp_path)
    assert len(passes) >= 4                       # the ISS passes a mid-latitude site several times a day
    for p in passes:
        assert p["rise"].tt < p["culminate"].tt < p["set"].tt
        assert pp.MIN_ELEVATION_DEG < p["peak_deg"] <= 90
        assert 0 < p["duration_min"] < 15
        # At rise and set the satellite sits on the 10 deg line.
        assert abs(pp.elevation_deg(sat, STATION, p["rise"]) - pp.MIN_ELEVATION_DEG) < 0.1
        assert abs(pp.elevation_deg(sat, STATION, p["set"]) - pp.MIN_ELEVATION_DEG) < 0.1
    rises = [p["rise"].tt for p in passes]
    assert rises == sorted(rises)


def test_pass_already_up_at_start_is_dropped(tmp_path):
    sat, passes = passes_over_two_days(tmp_path)
    mid = passes[0]["culminate"]
    later = pp.find_passes(sat, STATION, mid, sat.epoch + 2)
    # Same passes, minus the first; the search lands within a fraction of a second.
    assert len(later) == len(passes) - 1
    for a, b in zip(later, passes[1:]):
        assert abs(a["rise"] - b["rise"]) * 86400 < 1


def test_highest_pass_passes_checkpoint(tmp_path):
    sat, passes = passes_over_two_days(tmp_path)
    best = max(passes, key=lambda p: p["peak_deg"])
    seconds, el, az, dist = pp.sample_pass(sat, STATION, best)
    assert np.all(np.diff(seconds) == 1)
    assert abs(el.max() - best["peak_deg"]) < 0.05
    assert np.all((0 <= az) & (az < 360))
    assert 300 < dist.min() < 1500
    assert pp.checkpoint(seconds, el) == []


def test_checkpoint_flags_bad_passes():
    s = np.arange(0, 600)
    one_peak = 10 + 60 * np.sin(np.pi * s / s[-1])
    assert pp.checkpoint(s, one_peak) == []
    assert "lasts" in pp.checkpoint(s[:120], one_peak[:120] * 0 + np.hanning(120))[0]
    two_peaks = 10 + 30 * np.abs(np.sin(2 * np.pi * s / s[-1]))
    assert "changes direction" in pp.checkpoint(s, two_peaks)[0]
    falling = one_peak[::-1][300:]
    assert any("changes direction" in m for m in pp.checkpoint(np.arange(len(falling)) * 2, falling))


def test_table_and_plot(tmp_path):
    sat, passes = passes_over_two_days(tmp_path)
    table = pp.save_table(passes, tmp_path / "passes.csv", {"norad": 25544, "station_id": 7})
    rows = list(csv.DictReader(open(table)))
    assert len(rows) == len(passes)
    assert rows[0]["norad"] == "25544" and rows[0]["rise_utc"] == pp.utc(passes[0]["rise"])
    best = max(passes, key=lambda p: p["peak_deg"])
    out = pp.plot_pass(*pp.sample_pass(sat, STATION, best)[:3], best, "test", tmp_path / "pass.png")
    assert out.stat().st_size > 10_000
