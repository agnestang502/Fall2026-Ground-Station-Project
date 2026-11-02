"""
Week 2 of the satellite-pass simulation: put down a ground station and predict passes.

Places a real SatNOGS ground station with wgs84.latlon(), finds every pass of the
satellite over it in the next 7 days (above 10 deg elevation), prints a table of
rise, culmination and set times with each pass's peak elevation, then samples the
highest pass every second and plots its elevation against time and a sky plot.

Week 2 checkpoint (on the highest pass):
  * it lasts roughly 5 to 15 minutes, and
  * its elevation rises, peaks once and falls.

Usage:
    python predict_passes.py                         # default satellite and station
    python predict_passes.py --find-station          # pick the station that heard the
                                                     # satellite most in the last 7 days
    python predict_passes.py --station-id 1234       # any SatNOGS station
    python predict_passes.py --lat 52.2 --lon 0.1 --alt-m 20   # any location
    python predict_passes.py --compare-satnogs       # stretch: check against SatNOGS
    python predict_passes.py --start 2026-11-02T00:00 --days 3   # a chosen window (UTC)

The TLE comes from find_satellite.fetch_tle (cached in data/tle/). Station details
from SatNOGS Network are cached in data/stations/<id>.json and reused when offline.
"""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from skyfield.api import wgs84

import find_satellite as fs
from download_funcube import get_page

# --- The ground station (override on the command line) ------------------------
# Chosen with --find-station on 2026-11-02: the online station with the most good
# UmKA-1 observations in the previous 7 days (16), a 3-element UHF Yagi in Japan.
STATION_ID = 5049                # JF6FYI, Kitakyushu: +33.895, +130.840, 10 m

NETWORK_API = "https://network.satnogs.org/api"
STATION_DIR = Path(__file__).parent / "data" / "stations"   # git-ignored cache
PASS_DIR = Path(__file__).parent / "data" / "passes"
PLOT_DIR = Path(__file__).parent / "data" / "plots"

DAYS_AHEAD = 7                   # how far ahead to look for passes
MIN_ELEVATION_DEG = 10           # a pass counts from when it climbs above this
STEP_S = 1                       # sampling step for the plotted pass
DURATION_RANGE_MIN = (5, 15)     # Week 2 checkpoint: typical LEO pass length
REQUEST_PAUSE = 6                # seconds between SatNOGS Network requests
MAX_OBS_PAGES = 8                # pages of 25 observations read by --find-station

RISE, CULMINATE, SET = 0, 1, 2   # skyfield find_events() codes


# --- The station --------------------------------------------------------------

def satnogs_get(path, params):
    """One page from the SatNOGS Network API: (records, next page URL)."""
    return get_page(f"{NETWORK_API}/{path}/?{urlencode(dict(params, format='json'))}")


def fetch_station(station_id):
    """SatNOGS station record (id, name, lat, lng, altitude, min_horizon, ...).

    Saved to data/stations/<id>.json; the saved copy is used if SatNOGS can't be reached.
    """
    path = STATION_DIR / f"{station_id}.json"
    try:
        records, _ = satnogs_get("stations", {"id": station_id})
        record = next(r for r in records if r.get("id") == station_id)
    except (OSError, StopIteration) as err:
        if not path.exists():
            raise SystemExit(f"Could not get SatNOGS station {station_id}: {err!r}")
        print(f"WARNING: SatNOGS unreachable ({err!r}); using saved station {path}")
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


def busiest_stations(norad, days=7):
    """[(station_id, good observations)] for this satellite over the last `days`,
    busiest first, from up to MAX_OBS_PAGES pages of SatNOGS Network observations."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # The API ignores "satellite__norad_cat_id" (see download_funcube.py), so filter on
    # "norad_cat_id" and re-check every record's satellite, status and time.
    url = (f"{NETWORK_API}/observations/?"
           + urlencode({"format": "json", "norad_cat_id": norad, "status": "good", "start": since}))
    counts = Counter()
    for _ in range(MAX_OBS_PAGES):
        records, url = get_page(url)
        counts.update(obs["ground_station"] for obs in records
                      if obs.get("norad_cat_id") == norad and obs.get("status") == "good"
                      and obs.get("start", "") >= since[:-1] and obs.get("ground_station"))
        if not url:
            break
        time.sleep(REQUEST_PAUSE)
    return counts.most_common()


def pick_station(norad):
    """The online station with the most recent good observations of the satellite."""
    ranked = busiest_stations(norad)
    if not ranked:
        raise SystemExit(f"No good SatNOGS observations of NORAD {norad} in the last 7 days")
    print("Stations with the most good observations in the last 7 days:")
    for station_id, n in ranked[:10]:
        print(f"  station {station_id:>5}: {n} good obs")
    for station_id, n in ranked[:10]:
        time.sleep(REQUEST_PAUSE)
        record = fetch_station(station_id)
        if record.get("status") == "Online":
            return record
        print(f"  skipping station {station_id}: status {record.get('status')}")
    raise SystemExit("None of the ten busiest stations is online")


def make_station(lat, lon, alt_m):
    """The station as a skyfield position that turns with the Earth."""
    return wgs84.latlon(lat, lon, elevation_m=alt_m)


# --- Passes ---------------------------------------------------------------------

def find_passes(sat, station, t0, t1, min_el=MIN_ELEVATION_DEG):
    """Every complete pass between t0 and t1, as dicts with rise, culmination and set
    times (skyfield Times) and the peak elevation in degrees.

    A pass that is already up at t0, or still up at t1, is left out.
    """
    times, codes = sat.find_events(station, t0, t1, altitude_degrees=min_el)
    passes, current = [], None
    for t, code in zip(times, codes):
        if code == RISE:
            current = {"rise": t, "culminations": []}
        elif current is None:
            continue                       # the pass began before t0
        elif code == CULMINATE:
            current["culminations"].append(t)
        else:
            current["set"] = t
            passes.append(current)
            current = None
    for p in passes:
        # A pass can (rarely) culminate twice; keep the higher peak.
        peaks = [(elevation_deg(sat, station, t), t) for t in p.pop("culminations")]
        p["peak_deg"], p["culminate"] = max(peaks, key=lambda pk: pk[0])
        p["duration_min"] = (p["set"] - p["rise"]) * 1440
    return passes


def elevation_deg(sat, station, t):
    alt, _, _ = (sat - station).at(t).altaz()
    return alt.degrees


def sample_pass(sat, station, p, step_s=STEP_S):
    """Seconds since rise, elevation, azimuth (deg) and range (km), every step_s."""
    seconds = np.arange(0, (p["set"] - p["rise"]) * 86400 + step_s / 2, step_s)
    t = fs.ts.tt_jd(p["rise"].tt + seconds / 86400)
    alt, az, dist = (sat - station).at(t).altaz()
    return seconds, alt.degrees, az.degrees, dist.km


def checkpoint(seconds, elevation):
    """Return a list of failed checks (empty means the Week 2 checkpoint passes)."""
    problems = []
    minutes = (seconds[-1] - seconds[0]) / 60
    lo, hi = DURATION_RANGE_MIN
    if not lo <= minutes <= hi:
        problems.append(f"pass lasts {minutes:.1f} min, outside {lo}-{hi} min")
    # Rises, peaks once and falls: the slope changes sign from + to - exactly once.
    slope = np.sign(np.diff(elevation))
    slope = slope[slope != 0]
    turns = int(np.sum(np.diff(slope) != 0))
    if turns != 1 or slope[0] < 0:
        problems.append(f"elevation changes direction {turns} times (expected one peak)")
    return problems


def utc(t):
    return t.utc_strftime("%Y-%m-%d %H:%M:%S")


def print_table(passes):
    print(f"\n{'#':>3}  {'rise (UTC)':<19}  {'culminate':<8}  {'set':<8}  "
          f"{'minutes':>7}  {'peak elev':>9}")
    for i, p in enumerate(passes):
        print(f"{i:>3}  {utc(p['rise'])}  {p['culminate'].utc_strftime('%H:%M:%S')}  "
              f"{p['set'].utc_strftime('%H:%M:%S')}  {p['duration_min']:>7.1f}  "
              f"{p['peak_deg']:>8.1f}°")


def save_table(passes, path, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass", "rise_utc", "culminate_utc", "set_utc", "duration_min",
                    "peak_elevation_deg", *extra])
        for i, p in enumerate(passes):
            w.writerow([i, utc(p["rise"]), utc(p["culminate"]), utc(p["set"]),
                        f"{p['duration_min']:.2f}", f"{p['peak_deg']:.2f}", *extra.values()])
    return path


def plot_pass(seconds, elevation, azimuth, p, title, out_path):
    """Elevation against time, and a sky plot with north up and east to the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(seconds / 60, elevation, lw=2)
    peak = int(np.argmax(elevation))
    ax.plot(seconds[peak] / 60, elevation[peak], "o", color="tab:red",
            label=f"peak {elevation[peak]:.1f}° at {p['culminate'].utc_strftime('%H:%M:%S')}")
    ax.axhline(MIN_ELEVATION_DEG, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(f"Minutes after rise ({utc(p['rise'])} UTC)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Elevation")

    sky = fig.add_subplot(1, 2, 2, projection="polar")
    sky.set_theta_zero_location("N")     # north at the top
    sky.set_theta_direction(-1)          # azimuth increases clockwise, like a compass
    theta = np.radians(azimuth)
    sky.plot(theta, 90 - elevation, lw=2)
    sky.plot(theta[0], 90 - elevation[0], "go", label="rise")
    sky.plot(theta[peak], 90 - elevation[peak], "o", color="tab:red", label="peak")
    sky.plot(theta[-1], 90 - elevation[-1], "ks", label="set")
    sky.set_rlim(0, 90)
    sky.set_rticks([0, 30, 60, 90 - MIN_ELEVATION_DEG])
    sky.set_yticklabels(["90°", "60°", "30°", f"{MIN_ELEVATION_DEG}°"])
    sky.set_xticks(np.radians([0, 90, 180, 270]))
    sky.set_xticklabels(["N", "E", "S", "W"])
    sky.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12))
    sky.set_title("Sky plot (centre = overhead)")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --- Stretch: compare with SatNOGS ---------------------------------------------

def satnogs_observations(norad, station_id, t0, t1):
    """This station's SatNOGS observations of the satellite between t0 and t1
    (scheduled future ones included), each with start, end and max_altitude."""
    params = {"norad_cat_id": norad, "ground_station": station_id,
              "start": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": t1.strftime("%Y-%m-%dT%H:%M:%SZ")}
    records, _ = satnogs_get("observations", params)
    return [r for r in records
            if r.get("norad_cat_id") == norad and r.get("ground_station") == station_id
            and r.get("start") and r.get("end")]


def compare_with_satnogs(sat, station, record, norad):
    """Match SatNOGS's pass windows for this station with our own prediction, using
    the station's own horizon, and print the differences."""
    now = datetime.now(timezone.utc)
    obs = satnogs_observations(norad, record["id"], now - timedelta(days=2),
                               now + timedelta(days=DAYS_AHEAD))
    horizon = float(record.get("min_horizon") or 0)
    print(f"\nStretch: {len(obs)} SatNOGS observation(s) of this satellite at station "
          f"{record['id']} (its horizon: {horizon:g}°)")
    rows = []
    for o in sorted(obs, key=lambda o: o["start"]):
        start = datetime.fromisoformat(o["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(o["end"].replace("Z", "+00:00"))
        ours = find_passes(sat, station, fs.ts.from_datetime(start - timedelta(minutes=20)),
                           fs.ts.from_datetime(end + timedelta(minutes=20)), min_el=horizon)
        if not ours:
            print(f"  {o['start']}: no matching pass in our prediction")
            continue
        p = min(ours, key=lambda q: abs((q["rise"].utc_datetime() - start).total_seconds()))
        d_rise = (p["rise"].utc_datetime() - start).total_seconds()
        d_set = (p["set"].utc_datetime() - end).total_seconds()
        d_peak = p["peak_deg"] - float(o.get("max_altitude") or "nan")
        rows.append((d_rise, d_set))
        print(f"  {o['start']} ({o.get('status')}): rise {d_rise:+5.0f} s, set {d_set:+5.0f} s, "
              f"peak {d_peak:+.1f}° (SatNOGS {o.get('max_altitude')}°)")
    if rows:
        worst = max(abs(x) for r in rows for x in r)
        print(f"  largest difference: {worst:.0f} s "
              f"({'within' if worst <= 60 else 'more than'} a minute)")
    return rows


def main():
    p = argparse.ArgumentParser(description="Week 2: predict passes over a ground station.")
    p.add_argument("--norad", type=int, default=fs.NORAD_ID, help="NORAD catalog number")
    p.add_argument("--tle-file", help="read the TLE from this file instead of CelesTrak")
    p.add_argument("--station-id", type=int, default=STATION_ID, help="SatNOGS station ID")
    p.add_argument("--find-station", action="store_true",
                   help="use the online station that heard the satellite most in the last 7 days")
    p.add_argument("--lat", type=float, help="station latitude (deg); with --lon and --alt-m")
    p.add_argument("--lon", type=float, help="station longitude (deg, east positive)")
    p.add_argument("--alt-m", type=float, default=0, help="station altitude (m)")
    p.add_argument("--start", help="start of the window, UTC, e.g. 2026-11-02T00:00 (default: now)")
    p.add_argument("--days", type=float, default=DAYS_AHEAD, help="how many days ahead")
    p.add_argument("--compare-satnogs", action="store_true",
                   help="stretch: compare with this station's SatNOGS pass windows")
    args = p.parse_args()

    sat = fs.fetch_tle(args.norad, args.tle_file)
    if args.lat is not None and args.lon is not None:
        record = {"id": None, "name": "custom location", "lat": args.lat, "lng": args.lon,
                  "altitude": args.alt_m}
    elif args.find_station:
        record = pick_station(args.norad)
    elif args.station_id is not None:
        record = fetch_station(args.station_id)
    else:
        raise SystemExit("Choose a station: --find-station, --station-id or --lat/--lon")
    station = make_station(record["lat"], record["lng"], record["altitude"])

    now = fs.ts.now()
    if args.start:
        now = fs.ts.from_datetime(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc))
    print(f"Satellite      : {sat.name} (NORAD {sat.model.satnum})")
    print(f"TLE line 1     : {sat.line1}")
    print(f"TLE line 2     : {sat.line2}")
    print(f"TLE epoch      : {utc(sat.epoch)} UTC ({fs.tle_age_days(sat, now):.2f} days old)")
    print(f"Station        : {record['name']} (SatNOGS station {record['id']})")
    print(f"Location       : {record['lat']:+.4f} deg, {record['lng']:+.4f} deg, "
          f"{record['altitude']:.0f} m")
    print(f"Window         : {utc(now)} UTC + {args.days:g} days, above {MIN_ELEVATION_DEG}°")

    passes = find_passes(sat, station, now, now + args.days)
    if not passes:
        raise SystemExit("No passes above the minimum elevation in this window")
    print_table(passes)
    tag = f"{sat.model.satnum}_{record['id'] or 'custom'}"
    table = save_table(passes, PASS_DIR / f"passes_{tag}.csv", {
        "norad": sat.model.satnum, "tle_line1": sat.line1, "tle_line2": sat.line2,
        "station_id": record["id"], "lat": record["lat"], "lon": record["lng"],
        "alt_m": record["altitude"]})
    print(f"\n{len(passes)} passes, {sum(DURATION_RANGE_MIN[0] <= q['duration_min'] <= DURATION_RANGE_MIN[1] for q in passes)} "
          f"of them {DURATION_RANGE_MIN[0]}-{DURATION_RANGE_MIN[1]} min long; table saved to {table}")

    best = max(passes, key=lambda q: q["peak_deg"])
    seconds, elevation, azimuth, dist = sample_pass(sat, station, best)
    print(f"\nHighest pass   : #{passes.index(best)}, rise {utc(best['rise'])} UTC, "
          f"peak {elevation.max():.1f}°, {(seconds[-1]) / 60:.1f} min, "
          f"closest range {dist.min():.0f} km")
    print(f"Rise / set az  : {azimuth[0]:.0f}° -> {azimuth[-1]:.0f}°")
    problems = checkpoint(seconds, elevation)
    print("Checkpoint     : " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    where = record["name"] + (f" (SatNOGS station {record['id']})" if record["id"] else "")
    title = f"{sat.name} over {where}, {best['rise'].utc_strftime('%Y-%m-%d')}"
    out = plot_pass(seconds, elevation, azimuth, best, title, PLOT_DIR / f"pass_{tag}.png")
    print(f"Plot           : {out}")

    if args.compare_satnogs and record["id"] is not None:
        compare_with_satnogs(sat, station, record, args.norad)


if __name__ == "__main__":
    main()
