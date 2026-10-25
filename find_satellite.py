"""
Week 1 of the satellite-pass simulation: pick a satellite and find it.

Fetches the satellite's TLE from CelesTrak, prints how old it is, then prints
where the satellite is right now (latitude, longitude, altitude) and checks the
Week 1 checkpoint:
  * the altitude is between about 300 and 1,000 km, and
  * the latitude is no further from the equator than the orbit's inclination
    (or 180 deg minus it, for orbits tilted past 90 deg).

Usage:
    python find_satellite.py                          # default satellite (below)
    python find_satellite.py --norad 43017 --freq-hz 145880000
    python find_satellite.py --tle-file my.tle        # offline: use a saved TLE
    python find_satellite.py --ground-track           # stretch: plot one orbit

Every fetched TLE is also saved to data/tle/<norad>.tle, and if CelesTrak can't
be reached the script falls back to that saved copy (and says so).
"""
import argparse
import math
from pathlib import Path

from skyfield.api import EarthSatellite, load, wgs84

# --- The satellite we simulate (override on the command line) -----------------
# Chosen with pick_satellite.py on 2026-10-25: an active CW beacon on SatNOGS DB,
# on 70 cm, in a ~500 km sun-synchronous orbit, with 25+ good SatNOGS Network
# observations in the previous 7 days.
NORAD_ID = 57172                 # UmKA-1 (RS40S)
BEACON_HZ = 437_625_000          # "Mode U - CW" beacon, f_0 in Hz

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
TLE_DIR = Path(__file__).parent / "data" / "tle"   # git-ignored cache of fetched TLEs

ALT_RANGE_KM = (300, 1000)       # Week 1 checkpoint: typical low-Earth-orbit altitudes
LAT_SLACK_DEG = 0.5              # geodetic vs geocentric latitude differ by up to ~0.2 deg

ts = load.timescale()            # uses skyfield's built-in tables; no download needed


def fetch_tle(norad, tle_file=None):
    """Return the satellite as a skyfield EarthSatellite.

    With tle_file, read that file. Otherwise download a fresh TLE from CelesTrak
    (saved to data/tle/<norad>.tle), falling back to the saved copy when offline.
    """
    if tle_file is not None:
        path, url = Path(tle_file), None
    else:
        path, url = TLE_DIR / f"{norad}.tle", CELESTRAK_URL.format(norad=norad)
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sats = load.tle_file(url or str(path), filename=str(path), reload=url is not None)
    except OSError as err:
        if url is None or not path.exists():
            raise SystemExit(f"Could not get a TLE for NORAD {norad}: {err}")
        print(f"WARNING: CelesTrak unreachable ({err}); using saved TLE {path}")
        sats = load.tle_file(str(path))
    for sat in sats:
        if sat.model.satnum == norad:
            # Keep the raw lines: later weeks save them so every pass can be regenerated.
            sat.line1, sat.line2 = tle_lines(path, norad)
            return sat
    raise SystemExit(f"No TLE for NORAD {norad} in {url or path}")


def tle_lines(path, norad):
    """The two TLE lines for this NORAD number, exactly as they appear in the file."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    one = next(ln for ln in lines if ln.startswith("1 ") and int(ln[2:7]) == norad)
    two = next(ln for ln in lines if ln.startswith("2 ") and int(ln[2:7]) == norad)
    return one, two


def tle_age_days(sat, t):
    """Days between the TLE's epoch and time t (positive = TLE is older than t)."""
    return t - sat.epoch


def inclination_deg(sat):
    return math.degrees(sat.model.inclo)


def period_minutes(sat):
    """Orbital period from the TLE's mean motion (no_kozai is in radians/minute)."""
    return 2 * math.pi / sat.model.no_kozai


def where_is(sat, t):
    """Latitude (deg), longitude (deg) and altitude (km) of the point below sat at t."""
    sub = wgs84.subpoint(sat.at(t))
    return sub.latitude.degrees, sub.longitude.degrees, sub.elevation.km


def max_latitude_deg(sat):
    """Furthest a ground track gets from the equator: i, or 180 - i past 90 deg."""
    inc = inclination_deg(sat)
    return inc if inc <= 90 else 180 - inc


def checkpoint(sat, lat, alt_km):
    """Return a list of failed checks (empty means the Week 1 checkpoint passes)."""
    problems = []
    lo, hi = ALT_RANGE_KM
    if not lo <= alt_km <= hi:
        problems.append(f"altitude {alt_km:.0f} km is outside {lo}-{hi} km")
    if abs(lat) > max_latitude_deg(sat) + LAT_SLACK_DEG:
        problems.append(f"|latitude| {abs(lat):.2f} deg exceeds the orbit's "
                        f"limit of {max_latitude_deg(sat):.2f} deg")
    return problems


def plot_ground_track(sat, t0, out_path, step_s=30):
    """Stretch: plot the ground track for one full orbit starting at t0."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = np.arange(0, period_minutes(sat) * 60 + step_s, step_s)
    t = ts.tt_jd(t0.tt + seconds / 86400)
    sub = wgs84.subpoint(sat.at(t))
    lat, lon = sub.latitude.degrees, sub.longitude.degrees

    # Break the line where longitude wraps from +180 to -180, so it doesn't
    # draw a stripe across the whole map.
    lon_plot = lon.copy()
    lon_plot[1:][np.abs(np.diff(lon)) > 180] = np.nan

    limit = max_latitude_deg(sat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lon_plot, lat, lw=1.5)
    ax.plot(lon[0], lat[0], "go", label=f"start {t0.utc_strftime('%Y-%m-%d %H:%M')} UTC")
    for y in (limit, -limit):
        ax.axhline(y, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.3)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"{sat.name} ground track, one orbit ({period_minutes(sat):.1f} min); "
                 f"dashed = ±{limit:.1f}° latitude limit")
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

    print(f"Satellite      : {sat.name} (NORAD {sat.model.satnum})")
    print(f"Beacon f_0     : {args.freq_hz / 1e6:.3f} MHz")
    print(f"TLE line 1     : {sat.line1}")
    print(f"TLE line 2     : {sat.line2}")
    print(f"TLE epoch      : {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"TLE age        : {tle_age_days(sat, now):.2f} days")
    print(f"Inclination    : {inclination_deg(sat):.2f} deg")
    print(f"Period         : {period_minutes(sat):.1f} min")
    print(f"Now            : {now.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Latitude       : {lat:+.4f} deg")
    print(f"Longitude      : {lon:+.4f} deg")
    print(f"Altitude       : {alt:.1f} km")

    if tle_age_days(sat, now) > 14:
        print("WARNING: this TLE is over two weeks old; predictions may be off by minutes.")

    problems = checkpoint(sat, lat, alt)
    print("Checkpoint     : " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    if args.ground_track:
        out = Path(__file__).parent / "data" / "plots" / f"ground_track_{sat.model.satnum}.png"
        print(f"Ground track   : {plot_ground_track(sat, now, out)}")


if __name__ == "__main__":
    main()
