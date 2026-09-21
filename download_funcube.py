import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# --- Parameters (the only "magic" values; change these freely) ----------------
NORAD_ID = 39444                # FUNcube-1
WANT_STATUS = "good"            # only human-vetted "good" recordings have usable audio
TARGET_COUNT = 10               # how many distinct passes we want to download
WINDOW_DAYS = 3                 # they must all fall within this many days of each other
MIN_SEPARATION_MIN = 30         # min minutes between chosen passes, so we don't grab the
                                # same ~15-min pass twice (FUNcube-1's orbit is ~97 min;
                                # raise toward ~90 to force strictly different revolutions)

API_BASE = "https://network.satnogs.org/api"
DATA_DIR = Path(__file__).parent / "data"                    # git-ignored audio lands here
MANIFEST_PATH = Path(__file__).parent / "downloads_manifest.json"  # record of what we pulled

# The observation fields that hold the audio-file URL, in order of preference.
# 'payload' = the raw .ogg; 'archive_url' = the archived copy for older passes.
AUDIO_FIELDS = ("payload", "archive_url")

USER_AGENT = "funcube-orbit-project/0.3 (student ground-station project)"
REQUEST_TIMEOUT = 60            # seconds per HTTP request
REQUEST_PAUSE = 2               # seconds between requests (be polite: ~60/hour limit)
MAX_PAGES = 20                  # safety cap on API pages. FUNcube-1 is observed almost
                                # continuously (~150 good obs per 2.5 h), so we need to page
                                # back a few hours to collect 10 passes spaced 30 min apart.
                                # The loop stops early as soon as it has TARGET_COUNT.

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


def select_passes():
    """
    Page through good FUNcube-1 observations (newest first) and pick up to
    TARGET_COUNT that (a) have audio, (b) fall within WINDOW_DAYS of the newest
    such observation, and (c) start at least MIN_SEPARATION_MIN apart from every
    pass already chosen -- so each one is a distinct pass, not a repeat.
    Returns a list of dicts: {id, start, audio_field, audio_url}.
    """
    query = urlencode({"satellite__norad_cat_id": NORAD_ID, "status": WANT_STATUS})
    url = f"{API_BASE}/observations/?{query}"

    chosen = []
    chosen_starts = []           # start times of chosen passes, for the separation test
    anchor = None                # start time of the newest good-with-audio observation
    separation = timedelta(minutes=MIN_SEPARATION_MIN)
    pages = 0
    stop = False

    while url and not stop and pages < MAX_PAGES:
        observations, next_url = get_page(url)
        pages += 1

        for obs in observations:
            if obs.get("status") != WANT_STATUS:   # re-check ourselves; a stray slipped through before
                continue
            audio_field, audio_url = find_audio_url(obs)
            if not audio_url:
                continue

            start = parse_time(obs["start"])
            if anchor is None:
                anchor = start                      # window ends at the newest good pass
            if anchor - start > timedelta(days=WINDOW_DAYS):
                stop = True                          # gone past the window; stop paging
                break

            # Only keep it if it's far enough from every pass already chosen.
            if any(abs(start - s) < separation for s in chosen_starts):
                continue

            chosen.append({
                "id": obs["id"],
                "start": obs["start"],
                "audio_field": audio_field,
                "audio_url": audio_url,
            })
            chosen_starts.append(start)
            if len(chosen) >= TARGET_COUNT:
                stop = True
                break

        if not stop:
            url = next_url
            if url:
                time.sleep(REQUEST_PAUSE)          # polite pause between pages

    return chosen


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chosen = select_passes()

    if not chosen:
        print("\nFound no good FUNcube-1 observations with audio. "
              "Check NORAD_ID / the API, then try again.")
        return

    # Report the window we actually got, and flag if it is short of the target.
    times = sorted(parse_time(c["start"]) for c in chosen)
    span = times[-1] - times[0]
    print(f"\nSelected {len(chosen)} distinct passes spanning {span} "
          f"({times[0].date()} to {times[-1].date()}).")
    if len(chosen) < TARGET_COUNT:
        print(f"  NOTE: wanted {TARGET_COUNT} but only found {len(chosen)} distinct passes "
              f"within {WINDOW_DAYS} days at >= {MIN_SEPARATION_MIN} min apart. "
              f"Widen WINDOW_DAYS, lower MIN_SEPARATION_MIN, or accept fewer.")

    # Download each, skipping files we already have (safe to re-run).
    print()
    manifest = []
    for c in chosen:
        destination = DATA_DIR / f"obs_{c['id']}.ogg"
        if destination.exists():
            print(f"skip (already have) {destination.name}")
        else:
            download(c["audio_url"], destination)
            time.sleep(REQUEST_PAUSE)
        manifest.append({**c, "file": destination.name})

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    have = sum(1 for c in manifest if (DATA_DIR / c["file"]).exists())
    print(f"\nDone. {have} audio files in {DATA_DIR}/  |  manifest: {MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()