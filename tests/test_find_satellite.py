"""Offline checks for find_satellite.py, using the ISS example TLE from skyfield's docs."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import find_satellite as fs  # noqa: E402

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082
2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473
"""


def iss(tmp_path):
    path = tmp_path / "iss.tle"
    path.write_text(ISS_TLE)
    return fs.fetch_tle(25544, path)


def test_reads_tle_file_and_epoch(tmp_path):
    sat = iss(tmp_path)
    assert sat.name == "ISS (ZARYA)"
    assert sat.epoch.utc_strftime("%Y-%m-%d") == "2014-01-20"
    assert abs(fs.tle_age_days(sat, sat.epoch + 3.5) - 3.5) < 1e-9
    assert sat.line1 == ISS_TLE.splitlines()[1]
    assert sat.line2 == ISS_TLE.splitlines()[2]
    assert abs(fs.inclination_deg(sat) - 51.6498) < 1e-6
    assert 92 < fs.period_minutes(sat) < 93


def test_position_passes_checkpoint_over_a_day(tmp_path):
    sat = iss(tmp_path)
    for minutes in range(0, 24 * 60, 7):
        lat, lon, alt = fs.where_is(sat, sat.epoch + minutes / 1440)
        assert -180 <= lon <= 180
        assert fs.checkpoint(sat, lat, alt) == []


def test_checkpoint_flags_bad_values(tmp_path):
    sat = iss(tmp_path)
    assert "altitude" in fs.checkpoint(sat, 0, 36000)[0]
    assert "latitude" in fs.checkpoint(sat, 70, 400)[0]


def test_retrograde_latitude_limit():
    sun_sync = SimpleNamespace(model=SimpleNamespace(inclo=math.radians(97.6)))
    assert abs(fs.max_latitude_deg(sun_sync) - 82.4) < 1e-9


def test_wrong_norad_in_file_is_rejected(tmp_path):
    path = tmp_path / "iss.tle"
    path.write_text(ISS_TLE)
    try:
        fs.fetch_tle(39444, path)
    except SystemExit as err:
        assert "No TLE for NORAD 39444" in str(err)
    else:
        raise AssertionError("expected SystemExit")


def test_ground_track_plot(tmp_path):
    sat = iss(tmp_path)
    out = fs.plot_ground_track(sat, sat.epoch, tmp_path / "track.png")
    assert out.stat().st_size > 10_000
