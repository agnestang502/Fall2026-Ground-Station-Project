# Week 2: Pass metadata + automatic signal validation

This page explains, in plain language, what the new code does, why, and what it found
on the 10 FUNcube-1 passes we already downloaded.

## The big picture (why we're doing this)

Volunteer ground stations around the world record satellites as they fly overhead
(a "pass") and upload the recordings to [SatNOGS](https://network.satnogs.org).
Each station uses a **TLE** (a standard two-line summary of the satellite's orbit)
to predict where the satellite is. From that it predicts how the radio frequency
will shift (the **Doppler effect**, the same effect that changes an ambulance
siren's pitch as it passes) and corrects for it.

If the TLE were perfect, the corrected signal would sit at one fixed pitch. In
practice it doesn't. The small leftover drift (the **residual**) is the difference
between what was *expected* and what was *observed*. Later in the project we'll use
those residuals to estimate the satellite's real position and velocity.

For that we need three things for every pass:

1. the recording,
2. **where the ground station is** (latitude, longitude, altitude), and
3. **the TLE the station used**,

plus confidence that the recording **actually contains the satellite's signal**. If we
measure the residual on a recording that's only noise, we'll get garbage numbers
that look real.

## What changed

### 1. `download_funcube.py` now saves metadata

For every pass it now writes three kinds of file into `data/<NORAD ID>/`:

| File | What it holds |
|---|---|
| `obs_<id>.ogg` | the audio recording (same as before) |
| `obs_<id>.json` | **the complete SatNOGS record**, including `station_lat`, `station_lng`, `station_alt`, `tle0/1/2`, transmitter frequency, start/end time, max elevation, and more |
| `manifest.json` | one summary row per pass with the important fields, so you don't have to open every JSON file |

Other improvements:

* **Any satellite**: pass `--norad <ID>`. Each satellite gets its own folder, because we'll
  work with more than one satellite.
* **"Recent" now means recent**: by default it only takes passes that started in the
  **last 24 hours** (`--hours` changes that), counted back from *now*. The old version
  used a 3-day window.
* It warns you if a pass is missing the station location or TLE.
* When the download finishes it **automatically runs the signal check**, so nothing is
  done by hand. (`--skip-validation` turns this off.)

### 2. New `validate_signals.py`: does each recording really have a signal?

Think of the recording as a song. The script turns it into a **spectrogram**, a
picture with time on the x-axis and pitch (frequency) on the y-axis. Bright means
loud. Pure noise looks like TV static. A satellite shows up as a **thin bright line**,
and because of the residual Doppler that line bends slowly over the pass.

The check works in four steps:

1. **Find bright spots.** For each second of audio it measures the local "static
   level" and marks only the spots that are *much* brighter than it.
2. **Connect the dots.** Bright spots in consecutive seconds are joined into a
   *track* if the pitch changes only a little between them, the way a real signal
   drifts. Random noise spots don't line up, so they never form a long track.
3. **Judge the best track.** To count as a signal it must:
   * last at least **20 seconds**, be present in **at least half** the seconds it
     spans, and be clearly above the noise;
   * drift **slowly** (less than 1.5 Hz per second). SatNOGS already removed the
     predicted Doppler, so a fast-moving line is usually a *different* satellite
     passing through the same band;
   * be **smooth**, not jumpy;
   * **not** stay at exactly one pitch the whole time. A perfectly flat line is
     more likely interference from something near the station than the satellite.
4. **Quality checks.** The script also looks for audio dropouts (gaps of silence),
   very short recordings, SatNOGS's own "without-signal" rating, and missing
   metadata.

Each pass gets one of three verdicts:

| Verdict | Meaning | Used later? |
|---|---|---|
| `SIGNAL` | clear, well-behaved satellite-like track | **yes** |
| `UNCERTAIN` | something is there, but a warning sign is present | no |
| `NO_SIGNAL` | nothing convincing | no |

**It's strict on purpose.** As discussed in the meeting, a false "yes" (accepting
noise) would poison the orbit analysis. A false "no" only costs one pass, so
anything doubtful is rejected.

The script writes:

* `validation_report.csv`: one row per pass with the verdict, the reasons, signal
  strength (SNR), start/end frequency of the track, drift rate, duration, file size,
  and the station location;
* `validation_plots/obs_<id>.png`: the spectrogram with the detected track in red,
  so a human can double-check any verdict in a couple of seconds.

**Hook for the earlier machine-learning classifier.** The meeting mentioned
using the ML signal test that was already built. That model isn't in this repository,
so `external_classifier_vote()` in `validate_signals.py` is the place to plug it in.
It's wired so the model can only make the check **stricter**: if it says "no
signal", the pass becomes `UNCERTAIN`.

### 3. Tests (`tests/`)

We can't know the right answer for real recordings, so the tests use **fake
recordings where we do know it**: pure noise, noise plus a weak drifting tone, a tone
that's too short, a constant interference tone, a recording with gaps, and another
satellite sweeping across the band. Results:

* **Pure noise: 0 false alarms in 25 separate noise recordings** (5 in the test suite
  and 20 more in a separate run).
* A drifting tone is detected when it is at least about **5% of the noise amplitude**
  (roughly 5 dB above the noise in a 6 Hz slice). Below that, it's rejected.
* Constant tones, recordings with gaps, and a crossing satellite are all kept out of
  `SIGNAL`.
* The downloader's pass-picking logic is tested against a fake SatNOGS response.
  It skips passes that are too old, too close together, missing audio, or have the
  wrong status, and it saves the location and TLE.

Run them with `python -m pytest tests/`.

## Results on our 10 FUNcube-1 passes

| Pass | Verdict | Why |
|---|---|---|
| 15028454 | **SIGNAL** | smooth line from 510 to 398 Hz for 257 s, drifting −0.30 Hz/s, 22 dB above the noise |
| 15034145 | UNCERTAIN | line at exactly 1465 Hz for 63 s with zero drift, so it could be interference |
| 15026718, 15026767, 15030415, 15035384 | NO_SIGNAL | nothing above the noise |
| 15033065, 15034070, 15034244, 15034713 | NO_SIGNAL | only short fragments (3–13 s); 15034244 and 15033065 also have audio dropouts |

**So only 1 of 10 passes survives**, even though SatNOGS had rated all 10 as "good".

Example plots (red dots = detected track):

![Validated signal](img/example_signal.png)
*Pass 15028454: the slowly bending red line is the kind of residual drift we'll use
later. The steep curves are other signals sweeping through; the checker ignores them.*

![Uncertain](img/example_uncertain.png)
*Pass 15034145: a perfectly flat tone. It might be the satellite, but it might be local
interference, so it's rejected.*

![No signal](img/example_no_signal.png)
*Pass 15035384: static only.*

### What this tells us

* **We should download many more than 10 passes.** Expect roughly 1 in 10 (this is a
  very small sample) to be usable, so aim for 50 to 100 per satellite.
* **SatNOGS's "good" rating isn't enough on its own.** A "good" rating says the
  observation worked by SatNOGS's standards. It doesn't guarantee a signal clean
  enough to measure frequency drift.
* **Known limitation:** the checker looks for *thin* lines, because a thin line is
  what we need to measure frequency precisely. A weak, *wide* signal (like
  FUNcube-1's data modulation) may be missed. That produces false negatives, not
  false positives, which fits the "err on the side of no" rule.

## Missing metadata for the 10 existing passes

Those 10 files were downloaded with the old script, which didn't save metadata. From
this sandbox SatNOGS couldn't be reached (blocked by network policy), so I couldn't
fetch it. Running `python download_funcube.py` on your own computer will download
fresh passes with full metadata. The 10 old files have been moved to `data/39444/`
to match the new layout.

## Data size (for the storage question)

* Measured: **about 1 MB per minute of audio**. A 4–8 minute pass is **2–15 MB**, and our
  10 passes total **58 MB**.
* 100 passes per satellite ≈ 0.6 GB. Ten passes a day for a month ≈ 1.7 GB per satellite.
* GitHub isn't meant for this: those `.ogg` files are currently committed to git
  (58 MB), which makes the repo slow to clone and can't be undone without rewriting
  history.
* Options to look into:
  * **Don't store the audio ourselves.** SatNOGS already keeps every recording
    (its archive is on archive.org, which is what the `archive_url` field points to).
    We could keep only the small `manifest.json` and metadata, plus our results, and
    re-download audio on demand. Metadata is a few KB per pass and fits fine in
    GitHub.
  * **Zenodo** (CERN): free for research data, gives a citable DOI, and allows large
    uploads per record.
  * **Kaggle Datasets / Hugging Face Datasets**: free public dataset hosting.
  * **Google Drive** (the mentor's suggestion): simplest for sharing within the team.

  Check each service's current size limits before choosing.

## How to run everything

```bash
pip install -r requirements.txt
python download_funcube.py                 # download + validate, FUNcube-1, last 24 h
python download_funcube.py --count 50 --hours 48
python validate_signals.py data/39444      # re-check a folder on its own
python -m pytest tests/                    # run the tests
```
