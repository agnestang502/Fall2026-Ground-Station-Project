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

The checker looks for the satellite in two different ways, because it can show up
in the audio in two different ways.

**Way 1: a thin line.** This works in four steps:

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

**Way 2: "quiet bursts" (added after checking real recordings by hand).** Many
stations save their audio in a way where the satellite doesn't appear as a line at
all. Instead, whenever the satellite's signal arrives, **the background hiss suddenly
gets quieter** for a few seconds. (It's the same reason an FM radio goes quiet
between songs when it's locked onto a strong station.) FUNcube-1 sends its data in
bursts, so these recordings show a hiss that keeps dipping for 5–10 seconds at a time.

We confirmed this against the SatNOGS website's own picture (the "waterfall") of pass
15071219 (Argentina). The satellite's bursts there appear at about 160, 220, 280,
310 and 340 seconds, and the checker finds quiet bursts at exactly those times.

To count as a signal this way, the recording needs:
* **at least 3 separate bursts** (1 or 2 could be a fluke);
* each lasting **at least 4 seconds**, and clearly deeper than the recording's normal
  second-to-second wobble (and at least 0.8 dB);
* one-second blips are ignored, and so are slow changes over minutes (a station's
  volume drifting) and total silence (that's lost audio, not a signal).

**Finally, quality checks.** The script also looks for audio dropouts (gaps of total
silence), very short recordings, SatNOGS's own "without-signal" rating, and missing
metadata. Any of these turns a "yes" into "not sure".

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
  It also lists how many quiet bursts were found and when;
* `validation_plots/obs_<id>.png`: the spectrogram with the detected track in red
  on top, and underneath it the hiss level over time with the quiet bursts shaded
  red, so a human can double-check any verdict in a couple of seconds.

**Hook for the earlier machine-learning classifier.** The meeting mentioned
using the ML signal test that was already built. That model isn't in this repository,
so `external_classifier_vote()` in `validate_signals.py` is the place to plug it in.
It's wired so the model can only make the check **stricter**: if it says "no
signal", the pass becomes `UNCERTAIN`.

### 3. Tests (`tests/`)

We can't know the right answer for real recordings, so the tests use **fake
recordings where we do know it**: pure noise, noise plus a weak drifting tone, a tone
that's too short, a constant interference tone, a recording with gaps, and another
satellite sweeping across the band. For the quiet bursts: four real-looking bursts,
only two bursts, lots of one-second blips, and a slowly drifting volume. Results:

* **Pure noise: 0 false alarms in 25 separate noise recordings** (5 in the test suite
  and 20 more in a separate run).
* A drifting tone is detected when it is at least about **5% of the noise amplitude**
  (roughly 5 dB above the noise in a 6 Hz slice). Below that, it's rejected.
* Constant tones, recordings with gaps, and a crossing satellite are all kept out of
  `SIGNAL`.
* Four quiet bursts about 2 dB deep are accepted. Two bursts, one-second blips, and a
  slowly drifting volume are all rejected. Pure noise still gave **0 false alarms**
  in another 20 tries with the burst check switched on.
* The downloader's pass-picking logic is tested against a fake SatNOGS response.
  It skips passes that are too old, too close together, missing audio, or have the
  wrong status, and it saves the location and TLE.

Run them with `python -m pytest tests/`.

## Results

### First version (thin lines only), on 29 passes

Only **1 of 29** passes passed. But when we compared with the SatNOGS website's
pictures by hand, several "no" passes clearly had the satellite in them, as short
wide bursts. That's how we found Way 2 above. The hand-check did its job.

### Current version (lines + quiet bursts), on the 13 recordings available here

| Pass | Station | Verdict | Why |
|---|---|---|---|
| 15071219 | LW2DYB (Argentina) | **SIGNAL** | 8 quiet bursts, matching the SatNOGS waterfall |
| 15068778 | DL2SKY (Germany) | **SIGNAL** | 4 quiet bursts (61–71, 222–225, 288–295, 302–309 s) |
| 15034244 | (old file) | **SIGNAL** | 10 quiet bursts. *Not yet checked against its SatNOGS waterfall* |
| 15028454 | (old file) | **SIGNAL** | smooth line from 510 to 398 Hz for 257 s, drifting −0.30 Hz/s |
| 15034145 | (old file) | UNCERTAIN | a perfectly flat line, which could be interference |
| 15070276 | Berowra (Australia) | NO_SIGNAL | nothing: no line, no bursts (agrees with the hand-check) |
| 7 other old files | | NO_SIGNAL | nothing, or only 1–2 bursts / short fragments |

**4 of 13 now pass**, up from 1. The other 16 new recordings are on Agnes's computer.
Re-running the checker there will give the full count.

![Quiet bursts](img/example_bursts.png)
*Pass 15071219 (Argentina). Bottom panel: the hiss level. Each red band is a quiet
burst, which is when the satellite was transmitting. The two deepest ones (around
10 s and 310 s) also show the signal itself in the top panel.*

![Line signal](img/example_signal.png)
*Pass 15028454: a different station's audio, where the satellite does show up as a
thin, slowly bending line. The steep curves are other signals sweeping through; the
checker ignores them.*

![Uncertain](img/example_uncertain.png)
*Pass 15034145: a perfectly flat tone. It might be the satellite, but it might be local
interference, so it's rejected.*

![No signal](img/example_no_signal.png)
*Pass 15035384: static only.*

### What this tells us

* **Always check the checker against real examples.** The first version looked fine
  on fake recordings but missed most real signals. Keep spot-checking a few passes
  against the SatNOGS waterfall, especially for a new satellite or new stations.
* **SatNOGS's "good" rating isn't enough on its own.** Some "good" passes really are
  empty (like 15070276).
* **Some stations probably can't hear this satellite well.** Several stations in the
  list have "UHF" in their name, but FUNcube-1 transmits on VHF.
* **A heads-up for the next phase.** In "quiet burst" recordings we know *when* the
  satellite was heard, but the audio doesn't show its exact pitch the way a thin line
  does. To measure the frequency shift, we may need the SatNOGS waterfall data (which
  keeps pitch information) or a different way of processing the audio. That's worth
  discussing with the mentor.

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
