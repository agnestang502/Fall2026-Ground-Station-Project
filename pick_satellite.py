"""
Week 1 helper: find satellites with an active CW beacon to simulate.

Lists every transmitter on SatNOGS DB with mode CW that is marked active and
alive, on a satellite SatNOGS lists as alive, then for each one:
  * keeps it if it's on the 2 m or 70 cm amateur-satellite sub-band,
  * looks up its TLE on CelesTrak and works out its perigee and apogee altitude,
  * counts its "good" observations on SatNOGS Network in the last few days.

Candidates in low Earth orbit (300-1,000 km) are ranked by recent good
observations, and the ranking is saved to data/cw_candidates.json.

Usage:
    python pick_satellite.py               # last 7 days of observations
    python pick_satellite.py --days 3 --top 15
"""
import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DB_API = "https://db.satnogs.org/api"
NETWORK_API = "https://network.satnogs.org/api"
CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE"
OUT_PATH = Path(__file__).parent / "data" / "cw_candidates.json"

ALT_RANGE_KM = (300, 1000)       # same range as the Week 1 checkpoint
BANDS_HZ = ((144.0e6, 146.0e6),  # amateur-satellite sub-bands most SatNOGS stations
            (435.0e6, 438.0e6))  # listen on: 2 m (VHF) and 70 cm (UHF)
EARTH_RADIUS_KM = 6378.135       # WGS72, the radius SGP4 uses
MU_KM3_S2 = 398600.8             # WGS72 gravitational parameter
OBS_PAGE_SIZE = 25               # SatNOGS Network returns 25 observations per page
REQUEST_PAUSE = 6                # seconds between SatNOGS Network requests
MAX_RETRY_WAIT = 300             # longest we wait after a 429 Too Many Requests
USER_AGENT = "funcube-orbit-project/0.5 (student ground-station project)"


def get(url, params=None, retries=3):
    if params:
        url = f"{url}?{urlencode(params)}"
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=120) as resp:
                return resp.read().decode()
        except OSError as err:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            if isinstance(err, HTTPError) and err.code == 429:    # rate limited
                wait = min(int(err.headers.get("Retry-After") or 60), MAX_RETRY_WAIT)
            print(f"  retrying in {wait} s: {url} ({err})", flush=True)
            time.sleep(wait)


def cw_transmitters():
    """Active, alive CW transmitters on satellites SatNOGS lists as in orbit."""
    in_orbit = {s["norad_cat_id"]: s for s in json.loads(get(f"{DB_API}/satellites/",
                                                            {"format": "json"}))
                if s.get("status") == "in orbit" and s.get("norad_cat_id")}
    txs = json.loads(get(f"{DB_API}/transmitters/", {"format": "json"}))
    return [dict(t, sat_name=in_orbit[t["norad_cat_id"]]["name"],
                 reception=in_orbit[t["norad_cat_id"]].get("reception_status") or "Unknown")
            for t in txs
            if t.get("mode") == "CW" and t.get("status") == "active" and t.get("alive")
            and t.get("downlink_low") and t.get("norad_cat_id") in in_orbit]


def parse_tles(text):
    """{norad: (name, line1, line2)} from a three-line TLE file."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    tles = {}
    for i in range(len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            tles[int(lines[i + 1][2:7])] = (lines[i].strip(), lines[i + 1], lines[i + 2])
    return tles


def orbit_from_tle(line1, line2):
    """Inclination (deg), perigee and apogee altitude (km), TLE epoch (UTC)."""
    inc = float(line2[8:16])
    ecc = float("0." + line2[26:33].strip())
    revs_per_day = float(line2[52:63])
    n = revs_per_day * 2 * math.pi / 86400                     # rad/s
    a = (MU_KM3_S2 / n ** 2) ** (1 / 3)
    yy, doy = int(line1[18:20]), float(line1[20:32])
    epoch = (datetime(2000 + yy if yy < 57 else 1900 + yy, 1, 1, tzinfo=timezone.utc)
             + timedelta(days=doy - 1))
    return inc, a * (1 - ecc) - EARTH_RADIUS_KM, a * (1 + ecc) - EARTH_RADIUS_KM, epoch


def good_observations(norad, since):
    """Number of good observations since `since` (capped at one page, 25), or None
    if SatNOGS Network keeps refusing the request."""
    # The API silently ignores "satellite__norad_cat_id" (see download_funcube.py), so
    # filter on "norad_cat_id" and re-check every record's satellite, status and time.
    try:
        page = json.loads(get(f"{NETWORK_API}/observations/", {
            "format": "json", "norad_cat_id": norad, "status": "good",
            "start": since.strftime("%Y-%m-%dT%H:%M:%SZ")}))
    except OSError as err:
        print(f"  gave up on NORAD {norad}: {err}", flush=True)
        return None
    return sum(1 for obs in page
               if obs.get("norad_cat_id") == norad and obs.get("status") == "good"
               and obs.get("start", "") >= since.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    p = argparse.ArgumentParser(description="Rank satellites with active CW beacons.")
    p.add_argument("--days", type=float, default=7, help="observation window (days)")
    p.add_argument("--check", type=int, default=20,
                   help="how many satellites to look up on SatNOGS Network (rate-limited)")
    p.add_argument("--top", type=int, default=20, help="how many to print")
    args = p.parse_args()

    txs = cw_transmitters()
    print(f"{len(txs)} active CW transmitters on satellites in orbit (SatNOGS DB)")
    tles = parse_tles(get(CELESTRAK_ACTIVE))
    print(f"{len(tles)} TLEs from CelesTrak's active group")

    lo, hi = ALT_RANGE_KM
    rows = []
    for t in txs:
        norad = t["norad_cat_id"]
        if norad not in tles:
            continue
        name, l1, l2 = tles[norad]
        inc, perigee, apogee, epoch = orbit_from_tle(l1, l2)
        if not (lo <= perigee and apogee <= hi):
            continue
        if not any(b0 <= t["downlink_low"] <= b1 for b0, b1 in BANDS_HZ):
            continue
        rows.append({
            "norad": norad, "name": t["sat_name"], "tle_name": name,
            "transmitter": t["description"], "uuid": t["uuid"],
            "downlink_hz": t["downlink_low"], "service": t.get("service"),
            "reception_status": t["reception"],
            "inclination_deg": round(inc, 2), "perigee_km": round(perigee),
            "apogee_km": round(apogee), "tle_epoch": epoch.isoformat(timespec="seconds"),
            "tle_line1": l1, "tle_line2": l2,
            "good_obs_recent": None, "obs_window_days": args.days,
        })
    print(f"{len(rows)} of them in {lo}-{hi} km orbits on 2 m / 70 cm; SatNOGS reception status: "
          f"{dict(Counter(r['reception_status'] for r in rows))}")

    # SatNOGS Network allows only a few requests a minute, so look up the most
    # promising satellites first: a known reception status, then the freshest TLE.
    rows.sort(key=lambda r: r["tle_epoch"], reverse=True)
    rows.sort(key=lambda r: r["reception_status"] == "Unknown")
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    checked = {}
    for r in rows:
        if r["norad"] not in checked and len(checked) < args.check:
            time.sleep(REQUEST_PAUSE)
            checked[r["norad"]] = good_observations(r["norad"], since)
            print(f"  {r['norad']:>6} {r['name']}: {checked[r['norad']]} good obs", flush=True)
        r["good_obs_recent"] = checked.get(r["norad"])
    rows.sort(key=lambda r: (-1 if r["good_obs_recent"] is None else -r["good_obs_recent"],
                             -r["downlink_hz"]))

    print(f"\nCW transmitters in {lo}-{hi} km orbits, ranked by good observations "
          f"in the last {args.days:g} days ({OBS_PAGE_SIZE} = {OBS_PAGE_SIZE}+, - = not checked):")
    print(f"{'NORAD':>6}  {'satellite':<22} {'f_0 (MHz)':>10}  {'inc':>6}  "
          f"{'perigee-apogee':>15}  {'good obs':>8}  {'reception':<10} transmitter")
    for r in rows[:args.top]:
        obs = "-" if r["good_obs_recent"] is None else r["good_obs_recent"]
        print(f"{r['norad']:>6}  {r['name'][:22]:<22} {r['downlink_hz'] / 1e6:>10.4f}  "
              f"{r['inclination_deg']:>6.2f}  {r['perigee_km']:>6}-{r['apogee_km']:<5} km  "
              f"{obs:>8}  {r['reception_status'][:10]:<10} {r['transmitter']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
