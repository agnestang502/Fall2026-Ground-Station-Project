"""Offline checks for basilisk_pass.py, using the ISS example TLE from skyfield's docs.

Skipped when Basilisk isn't installed (pip install bsk). The short simulation test needs
Basilisk's GGM03S gravity file, which it downloads from GitHub the first time."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Basilisk")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import basilisk_pass as bp  # noqa: E402
import doppler_pass as dp  # noqa: E402
import find_satellite as fs  # noqa: E402
import predict_passes as pp  # noqa: E402
from test_find_satellite import iss  # noqa: E402

RECORD = {"id": None, "name": "test", "lat": 40.0, "lng": -75.0, "altitude": 100}
F0 = 437_625_000


def test_tle_state_matches_skyfield(tmp_path):
    sat = iss(tmp_path)
    r, v = bp.tle_state(tmp_path / "iss.tle", 25544)
    g = sat.at(sat.epoch)
    assert np.linalg.norm(r - g.position.m) < bp.START_MATCH_M
    assert np.linalg.norm(v - g.velocity.km_per_s * 1e3) < 0.01


def test_earth_rotation_rate_matches_finite_difference(tmp_path):
    t0 = iss(tmp_path).epoch
    t = dp.times_after(t0, np.array([-0.5, 0.0, 0.5]))
    R, R_dot = bp.earth_rotation(t)
    assert np.allclose(R[:, :, 1] @ R[:, :, 1].T, np.eye(3), atol=1e-12)
    assert np.allclose(R_dot[:, :, 1], R[:, :, 2] - R[:, :, 0], atol=1e-11)


def test_sun_near_the_equinox():
    t = fs.ts.from_datetime(datetime(2026, 3, 20, 14, 46, tzinfo=timezone.utc))
    sun = bp.sun_position_m(t)[:, 0]
    assert abs(np.linalg.norm(sun) / bp.AU_M - 0.996) < 0.002
    assert np.degrees(np.arccos(sun[0] / np.linalg.norm(sun))) < 0.1   # straight along +x


def test_ecef_to_geodetic_round_trip():
    pos = pp.make_station(33.895, 130.84, 500e3).itrs_xyz.m
    lat, lon, alt = bp.ecef_to_geodetic(pos)
    assert abs(lat - 33.895) < 1e-7 and abs(lon - 130.84) < 1e-7 and abs(alt - 500e3) < 0.01


def test_passes_from_elevation():
    s = np.arange(0.0, 900.0)
    elev = 40 * np.sin(2 * np.pi * (s - 100) / 600)       # one full pass, then one still up at the end
    passes = bp.passes_from_elevation(s, elev)
    assert len(passes) == 1                               # the second rise never sets
    p = passes[0]
    expected_rise = 100 + 600 * np.arcsin(10 / 40) / (2 * np.pi)
    assert abs(p["rise_s"] - expected_rise) < 0.05
    assert abs(p["set_s"] - (400 - (expected_rise - 100))) < 0.05
    assert p["peak_s"] == 250 and abs(p["peak_deg"] - 40) < 1e-9


def test_drag_lag_from_the_week3_tle(tmp_path):
    path = tmp_path / "umka.tle"
    path.write_text("UMKA 1 (RS40S)\n"
                    "1 57172U 23091G   26312.31862081  .00010117  00000+0  44267-3 0  9995\n"
                    "2 57172  97.4878  10.4506 0011623 343.3809  16.7045 15.22190993185925\n")
    sat = fs.fetch_tle(57172, path)
    assert abs(bp.drag_lag_m(sat, 5.21 * 86400) / 1e3 - 118.7) < 0.5
    assert bp.drag_lag_m(sat, 2 * 3600) < 0.1 * 1e3 * 2   # ~0.2 km after two hours


def test_short_run_agrees_with_sgp4(tmp_path):
    """Ten minutes of Basilisk: it stays within a few hundred meters of SGP4, and
    groundLocation's range-rate matches the Week 3 formula on Basilisk's own orbit."""
    sat = iss(tmp_path)
    r0, v0 = bp.tle_state(tmp_path / "iss.tle", 25544)
    parts = bp.build_sim(sat, RECORD, r0, v0, F0, degree=4)
    parts["sim"].InitializeSimulation()
    bp.run_until(parts, 600)
    res = bp.results(parts)
    t = dp.times_after(sat.epoch, res["seconds"])
    drift = np.linalg.norm(res["r"] - sat.at(t).position.m, axis=0)
    assert drift[0] < bp.START_MATCH_M and drift.max() < 500

    R, R_dot = bp.earth_rotation(t)
    r_sta = np.einsum("jin,j->in", R, parts["station_xyz"])
    v_sta = np.einsum("jin,j->in", R_dot, parts["station_xyz"])
    rng, rate = dp.range_rate_dot(res["r"], res["v"], r_sta, v_sta)
    assert np.allclose(res["range_m"], rng, atol=1e-3)
    assert np.max(np.abs(res["range_rate"] - rate)) < bp.RANGE_RATE_MATCH_M_S

    elev = bp.geodetic_elevation_deg(res["r"], R, parts["station_xyz"], RECORD["lat"], RECORD["lng"])
    station = pp.make_station(RECORD["lat"], RECORD["lng"], RECORD["altitude"])
    assert np.max(np.abs(elev - pp.elevation_deg(sat, station, t))) < 0.2


def test_vizard_file_is_written(tmp_path):
    sat = iss(tmp_path)
    r0, v0 = bp.tle_state(tmp_path / "iss.tle", 25544)
    out = tmp_path / "pass.bin"
    parts = bp.build_sim(sat, RECORD, r0, v0, F0, viz_file=out, degree=2)
    parts["sim"].InitializeSimulation()
    parts["sim"].enableTask("viz")
    bp.run_until(parts, 60)
    assert out.stat().st_size > 1000
