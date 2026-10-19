"""
Download recent SatNOGS observations (audio + metadata) for one satellite.

For every pass we keep:
  * the audio recording (.ogg)                  -> data/<norad>/obs_<id>.ogg
  * the SatNOGS waterfall image (.png)          -> data/<norad>/obs_<id>_waterfall.png
    (used by the CNN signal screen in waterfall_cnn.py)
  * the FULL observation record from SatNOGS    -> data/<norad>/obs_<id>.json
    (ground-station location, the TLE the station used, frequencies, etc.)
  * one summary row per pass in a manifest      -> data/<norad>/manifest.json

Usage:
    python download_funcube.py                    # FUNcube-1, last 24 h, 10 passes
    python download_funcube.py --norad 43017      # a different satellite
    python download_funcube.py --hours 48 --count 20
    python download_funcube.py --skip-validation  # download only

After downloading, every pass is automatically checked for a real signal by
validate_signals.py (see that file / docs/signal_validation.md).
"""
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# --- Defaults (all can be overridden on the command line) ---------------------
NORAD_ID = 39444                # FUNcube-1
WANT_STATUS = "good"            # SatNOGS vetting status; use "any" to skip this filter
TARGET_COUNT = 10               # how many distinct passes we want (a minimum, ideally)
WINDOW_HOURS = 24               # only passes that started in the last this-many hours
MIN_SEPARATION_MIN = 30         # min minutes between chosen passes, so we don't grab the
                                # same ~15-min pass twice (FUNcube-1's orbit is ~97 min;
                                # raise toward ~90 to force strictly different revolutions)

API_BASE = "https://network.satnogs.org/api"
DATA_ROOT = Path(__file__).parent / "data"   # git-ignored; one sub-folder per NORAD ID

# The observation fields that hold the audio-file URL, in order of preference.
# 'payload' = the raw .ogg; 'archive_url' = the archived copy for older passes.
AUDIO_FIELDS = ("payload", "archive_url")

# Metadata we copy into the manifest summary. The complete record is always saved
# to obs_<id>.json as well, so nothing is lost if a field here is renamed/missing.
SUMMARY_FIELDS = (
    "start", "end", "status", "vetted_status", "waterfall_status",
    "ground_station", "station_name", "station_lat", "station_lng", "station_alt",
    "tle0", "tle1", "tle2",
    "transmitter_uuid", "transmitter_mode", "transmitter_baud",
    "transmitter_downlink_low", "transmitter_downlink_drift",
    "center_frequency", "observation_frequency",
    "rise_azimuth", "set_azimuth", "max_altitude",
    "client_version",
)

USER_AGENT = "funcube-orbit-project/0.4 (student ground-station project)"
REQUEST_TIMEOUT = 60            # seconds per HTTP request
REQUEST_PAUSE = 2               # seconds between requests (be polite: ~60/hour limit)
MAX_PAGES = 20                  # safety cap on API pages; the loop usually stops sooner


def parse_next_link(link_header):
    """Pull the rel="next" URL out of an HTTP Link header, or return None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        pieces = part.split(";")
        if len(pieces) < 2:
            continue
        url = pieces[0].strip().strip("<>").strip()
        if 'rel="next"' in pieces[1]:
            return url
    return None


def get_page(url, retries=3):
    """
    GET one page of observations. Returns (observations_list, next_page_url).
    Retries politely on a 429 rate-limit response.
    """
    print(f"GET {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                observations = json.load(response)
                next_url = parse_next_link(response.headers.get("Link"))
                return observations, next_url
        except HTTPError as error:
            if error.code == 429 and attempt < retries - 1:
                wait = REQUEST_PAUSE * (2 ** attempt)
                print(f"  rate-limited (429); waiting {wait}s then retrying...")
                time.sleep(wait)
                continue
            raise
    return [], None


def find_audio_url(observation):
    """
    Return (field_name, url) for this observation's audio file, or (None, None).
    Checks 'payload' then 'archive_url', then any string value containing '.ogg'.
    """
    for field_name in AUDIO_FIELDS:
        value = observation.get(field_name)
        if isinstance(value, str) and value.strip():
            return field_name, value
    for field_name, value in observation.items():
        if isinstance(value, str) and ".ogg" in value.lower():
            return field_name, value
    return None, None


def parse_time(text):
    """Parse an API timestamp like '2026-09-21T00:25:36Z' into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def download(url, destination):
    """Download a URL to a file on disk."""
    print(f"  downloading -> {destination.name}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response, open(destination, "wb") as out:
        out.write(response.read())
    print(f"    done ({destination.stat().st_size} bytes)")


def missing_metadata(obs):
    """List the metadata we need for orbit work (location + TLE) that this record lacks."""
    needed = ("station_lat", "station_lng", "station_alt", "tle1", "tle2")
    return [name for name in needed if obs.get(name) in (None, "")]


def select_passes(norad_id, status, count, window_hours, separation_min):
    """
    Page through observations of this satellite (newest first) and pick up to
    `count` that (a) have audio, (b) started within the last `window_hours`, and
    (c) start at least `separation_min` minutes from every pass already chosen,
    so each one is a distinct pass, not a repeat.
    Returns the full SatNOGS observation records (dicts) that were chosen.
    """
    # Note: the API silently ignores "satellite__norad_cat_id"; "norad_cat_id" is the
    # filter that works. Each record's ID is also checked below, to be safe.
    params = {"norad_cat_id": norad_id}
    if status != "any":
        params["status"] = status
    url = f"{API_BASE}/observations/?{urlencode(params)}"

    now = datetime.now(timezone.utc)
    oldest_allowed = now - timedelta(hours=window_hours)
    separation = timedelta(minutes=separation_min)
    chosen = []
    pages = 0
    stop = False

    while url and not stop and pages < MAX_PAGES:
        observations, next_url = get_page(url)
        pages += 1

        for obs in observations:
            if obs.get("norad_cat_id") not in (None, norad_id):
                continue                             # a different satellite
            if status != "any" and obs.get("status") != status:
                continue                             # re-check; a stray slipped through before
            start = parse_time(obs["start"])
            if start > now:
                continue                             # scheduled future pass, no audio yet
            if start < oldest_allowed:
                stop = True                          # gone past the window; stop paging
                break
            if not find_audio_url(obs)[1]:
                continue
            if any(abs(start - parse_time(c["start"])) < separation for c in chosen):
                continue

            chosen.append(obs)
            if len(chosen) >= count:
                stop = True
                break

        if not stop:
            url = next_url
            if url:
                time.sleep(REQUEST_PAUSE)          # polite pause between pages

    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--norad", type=int, default=NORAD_ID, help="satellite NORAD ID")
    parser.add_argument("--count", type=int, default=TARGET_COUNT, help="passes to download")
    parser.add_argument("--hours", type=float, default=WINDOW_HOURS, help="look-back window")
    parser.add_argument("--status", default=WANT_STATUS,
                        help='SatNOGS status filter: good, bad, failed, unknown or "any"')
    parser.add_argument("--separation", type=float, default=MIN_SEPARATION_MIN,
                        help="minimum minutes between chosen passes")
    parser.add_argument("--skip-validation", action="store_true",
                        help="don't run the signal check after downloading")
    args = parser.parse_args()

    out_dir = DATA_ROOT / str(args.norad)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = select_passes(args.norad, args.status, args.count, args.hours, args.separation)
    if not chosen:
        print(f"\nFound no observations of NORAD {args.norad} with audio in the last "
              f"{args.hours:g} h. Try a larger --hours or --status any.")
        return

    times = sorted(parse_time(o["start"]) for o in chosen)
    print(f"\nSelected {len(chosen)} distinct passes from {times[0]} to {times[-1]}.")
    if len(chosen) < args.count:
        print(f"  NOTE: wanted {args.count} but only found {len(chosen)} within "
              f"{args.hours:g} h at >= {args.separation:g} min apart. "
              f"Widen --hours, lower --separation, or accept fewer.")

    print()
    manifest = []
    for obs in chosen:
        obs_id = obs["id"]
        audio_field, audio_url = find_audio_url(obs)
        audio_path = out_dir / f"obs_{obs_id}.ogg"
        meta_path = out_dir / f"obs_{obs_id}.json"
        waterfall_path = out_dir / f"obs_{obs_id}_waterfall.png"

        # Always (re)write the metadata: it is small and may have been updated.
        meta_path.write_text(json.dumps(obs, indent=2))
        if audio_path.exists():
            print(f"skip (already have) {audio_path.name}")
        else:
            download(audio_url, audio_path)
            time.sleep(REQUEST_PAUSE)
        if obs.get("waterfall") and not waterfall_path.exists():
            download(obs["waterfall"], waterfall_path)
            time.sleep(REQUEST_PAUSE)

        gaps = missing_metadata(obs)
        if gaps:
            print(f"  WARNING obs {obs_id}: metadata missing {', '.join(gaps)}")

        manifest.append({
            "id": obs_id,
            "norad_id": args.norad,
            "audio_file": audio_path.name,
            "metadata_file": meta_path.name,
            "waterfall_file": waterfall_path.name if waterfall_path.exists() else None,
            "audio_field": audio_field,
            "audio_url": audio_url,
            **{name: obs.get(name) for name in SUMMARY_FIELDS},
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    stations = {m["station_name"] or m["ground_station"] for m in manifest}
    print(f"\nDone. {len(manifest)} passes from {len(stations)} ground station(s) "
          f"in {out_dir}/  |  manifest: {manifest_path}")
    if args.skip_validation:
        print(f"Validate later with: python validate_signals.py {out_dir}")
    else:
        from validate_signals import validate_folder   # needs numpy/scipy/soundfile
        print("\nChecking each pass for a real signal...\n")
        validate_folder(out_dir)


if __name__ == "__main__":
    main()
