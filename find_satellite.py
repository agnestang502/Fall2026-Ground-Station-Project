"""
Week 1 of the satellite-pass simulation: pick a satellite and find it.

Fetches the satellite's TLE from CelesTrak, prints how old it is, then prints
where the satellite is right now and checks the Week 1 checkpoint:
  * the altitude is between about 300 and 1,000 km, and
  * the latitude is no further from the equator than the orbit's inclination
    (or 180 deg minus it, for orbits tilted past 90 deg).

Usage:
    python find_satellite.py                    # UmKA-1 (below)
    python find_satellite.py --ground-track     # also plot one orbit (stretch goal)
    python find_satellite.py --norad 59112 --freq-hz 145840000
"""
import argparse
import math
from pathlib import Path

from skyfield.api import load, wgs84

# The satellite we simulate, chosen on SatNOGS DB (see docs/week1_find_satellite.md).
NORAD_ID = 57172                 # UmKA-1 (RS40S)
BEACON_HZ = 437_625_000          # its CW beacon, f_0 in Hz

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
TLE_DIR = Path(__file__).parent / "data" / "tle"   # git-ignored
ALT_RANGE_KM = (300, 1000)
LAT_SLACK_DEG = 0.5              # geodetic vs geocentric latitude differ by up to ~0.2 deg

ts = load.timescale()


def fetch_tle(norad, tle_file=None):
    """The satellite as a skyfield EarthSatellite, from tle_file or a fresh CelesTrak TLE."""
    path = Path(tle_file) if tle_file else TLE_DIR / f"{norad}.tle"
    if not tle_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        load.download(CELESTRAK_URL.format(norad=norad), filename=str(path))
    sat = next((s for s in load.tle_file(str(path)) if s.model.satnum == norad), None)
    if sat is None:
        raise SystemExit(f"No TLE for NORAD {norad} in {path}")
    # Keep the raw lines: Week 5 saves them so every pass can be regenerated.
    lines = path.read_text().splitlines()
    sat.line1, sat.line2 = (next(ln.strip() for ln in lines if ln.startswith(f"{n} {norad:05d}"))
                            for n in (1, 2))
    return sat


def inclination_deg(sat):
    return math.degrees(sat.model.inclo)


def period_minutes(sat):
    return 2 * math.pi / sat.model.no_kozai       # no_kozai is in radians/minute


def max_latitude_deg(sat):
    """Furthest a ground track gets from the equator: i, or 180 - i past 90 deg."""
    inc = inclination_deg(sat)
    return inc if inc <= 90 else 180 - inc


def where_is(sat, t):
    """Latitude (deg), longitude (deg) and altitude (km) of the point below sat at t."""
    sub = wgs84.subpoint(sat.at(t))
    return sub.latitude.degrees, sub.longitude.degrees, sub.elevation.km


def checkpoint(sat, lat, alt_km):
    """List of failed checks (empty means the Week 1 checkpoint passes)."""
    problems = []
    if not ALT_RANGE_KM[0] <= alt_km <= ALT_RANGE_KM[1]:
        problems.append(f"altitude {alt_km:.0f} km is outside {ALT_RANGE_KM[0]}-{ALT_RANGE_KM[1]} km")
    if abs(lat) > max_latitude_deg(sat) + LAT_SLACK_DEG:
        problems.append(f"|latitude| {abs(lat):.2f} deg exceeds {max_latitude_deg(sat):.2f} deg")
    return problems


def plot_ground_track(sat, t0, out_path, step_s=30):
    """Stretch goal: the ground track for one full orbit starting at t0."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = np.arange(0, period_minutes(sat) * 60 + step_s, step_s)
    sub = wgs84.subpoint(sat.at(ts.tt_jd(t0.tt + seconds / 86400)))
    lat, lon = sub.latitude.degrees, sub.longitude.degrees
    lon_plot = lon.copy()
    lon_plot[1:][np.abs(np.diff(lon)) > 180] = np.nan   # don't draw across the 180 deg seam

    limit = max_latitude_deg(sat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lon_plot, lat, lw=1.5)
    ax.plot(lon[0], lat[0], "go", label=f"start {t0.utc_strftime('%Y-%m-%d %H:%M')} UTC")
    for y in (limit, -limit):
        ax.axhline(y, color="gray", ls="--", lw=0.8)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xticks=range(-180, 181, 30),
           yticks=range(-90, 91, 30), xlabel="Longitude (deg)", ylabel="Latitude (deg)",
           title=f"{sat.name} ground track, one orbit ({period_minutes(sat):.1f} min); "
                 f"dashed = ±{limit:.1f}° latitude limit")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Week 1: fetch a TLE and find the satellite now.")
    p.add_argument("--norad", type=int, default=NORAD_ID, help="NORAD catalog number")
    p.add_argument("--freq-hz", type=float, default=BEACON_HZ, help="beacon frequency f_0 (Hz)")
    p.add_argument("--tle-file", help="read the TLE from this file instead of CelesTrak")
    p.add_argument("--ground-track", action="store_true", help="plot one orbit's ground track")
    args = p.parse_args()

    sat = fetch_tle(args.norad, args.tle_file)
    now = ts.now()
    lat, lon, alt = where_is(sat, now)
    age = now - sat.epoch

    print(f"Satellite   : {sat.name} (NORAD {args.norad}), f_0 = {args.freq_hz / 1e6:.3f} MHz")
    print(f"TLE         : {sat.line1}\n              {sat.line2}")
    print(f"TLE epoch   : {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC ({age:.2f} days old)")
    print(f"Orbit       : inclination {inclination_deg(sat):.2f} deg, "
          f"period {period_minutes(sat):.1f} min")
    print(f"Now         : {now.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Position    : lat {lat:+.4f} deg, lon {lon:+.4f} deg, alt {alt:.1f} km")
    if age > 14:
        print("WARNING: this TLE is over two weeks old; predictions may be off by minutes.")
    problems = checkpoint(sat, lat, alt)
    print("Checkpoint  : " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    if args.ground_track:
        out = Path(__file__).parent / "data" / "plots" / f"ground_track_{args.norad}.png"
        print(f"Ground track: {plot_ground_track(sat, now, out)}")


if __name__ == "__main__":
    main()
