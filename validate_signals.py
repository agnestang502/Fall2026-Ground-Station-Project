"""
Validate downloaded SatNOGS passes: is there really a satellite signal in each recording?

For every obs_<id>.ogg in a folder (made by download_funcube.py) this script:
  0. screens the SatNOGS waterfall image (obs_<id>_waterfall.png) with the CNN
     classifier in waterfall_cnn.py; a low score rejects the pass immediately,
  1. turns the audio into a spectrogram (signal strength vs. frequency vs. time),
  2. looks for signs of the satellite after Doppler correction:
     a) a steady ~1 kHz-wide band (FUNcube-1's BPSK telemetry),
     b) a narrow signal that stands well above the noise AND forms a continuous,
        slowly drifting track for a good stretch of the pass, or
     c) repeated "quiet bursts": several seconds at a time when the background hiss
        drops. Many stations record FM-demodulated audio, and in that audio a
        strong signal shows up as the hiss going quiet rather than as a line
        (checked against the SatNOGS waterfall of observation 15071219),
  3. runs quality checks (dropouts, very short recording, a constant tone that is
     probably local interference rather than the satellite),
  4. checks the metadata we need later (ground-station location and TLE),
and writes one verdict per pass: SIGNAL, NO_SIGNAL or UNCERTAIN.

It is deliberately conservative: only SIGNAL counts as usable. Anything doubtful is
UNCERTAIN, because a false "yes, there's a signal" would put bad data into the orbit
analysis, while a false "no" only costs us one pass.

Usage:
    python validate_signals.py data/39444            # writes report + plots there
    python validate_signals.py data/39444 --no-plots
    python validate_signals.py data/39444 --no-cnn   # audio check only
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

import waterfall_cnn
from waterfall_cnn import CNN_THRESHOLD

# --- Detection settings (tuned to be strict; see docs/signal_validation.md) ----
FREQ_RES_HZ = 6.0          # target spectrogram frequency resolution
FRAME_SEC = 1.0            # spectrogram columns are averaged into ~1 s frames
F_MIN_HZ = 100.0           # ignore DC / very low audio frequencies
NOISE_WINDOW_HZ = 2000.0   # width of the running median used as the "noise floor"
PEAKS_PER_FRAME = 3        # strongest candidate peaks kept in each frame
PEAK_SPACING_HZ = 50.0     # candidate peaks in one frame must be this far apart
Z_HIT = 6.0                # a peak must be this many noise std-devs above the floor
MAX_DRIFT_HZ_PER_S = 15.0  # how fast a track may move (residual Doppler is slow)
FREQ_TOL_HZ = 12.0         # plus this much jitter per step
MAX_GAP_FRAMES = 4         # a track may fade out for this many frames and continue

MIN_TRACK_HITS = 20        # need >= 20 s of detections on one track...
MIN_TRACK_FILL = 0.5       # ...covering >= 50% of the frames the track spans
MIN_MEDIAN_Z = 7.0         # ...with a typical strength well above the threshold

# Wide-signal detector. FUNcube-1's BPSK telemetry is not a thin line in the audio: it
# is a ~1 kHz-wide band that stays at one audio frequency (SatNOGS removes the Doppler).
# Peaks inside such a band jump around, so the thin-line tracker above misses it.
BAND_BIN_HZ = 50.0         # coarse frequency bins for this detector
BAND_WIDTH_HZ = 800.0      # average the power over a band this wide...
BAND_BG_HZ = 6000.0        # ...and compare it with the median over this much spectrum
BAND_Z_HIT = 6.0           # a frame counts if its best band is this many noise std-devs up
BAND_FREQ_TOL_HZ = 400.0   # and sits within this much of the band's usual frequency
BAND_MIN_FILL = 0.5        # SIGNAL: present in >= 50% of the recording...
BAND_MIN_EXCESS_DB = 3.0   # ...and typically >= 3 dB above the surrounding noise
BAND_WEAK_FILL = 0.3       # weaker than that but above these -> UNCERTAIN
BAND_WEAK_EXCESS_DB = 1.5

MAX_ACCEPT_DRIFT = 1.5     # Hz/s: SatNOGS already removes the predicted Doppler, so the
                           # target's leftover drift is small; faster usually means the
                           # tail of a different (uncorrected) satellite crossing by
MAX_SCATTER_HZ = 8.0       # track wobble around a smooth curve; jumpy = likely not ours
SPUR_MAX_DRIFT = 0.02      # Hz/s: a track whose frequency never moves could just as
                           # well be local interference, so it is not trusted
QUIET_MIN_DB = 0.8         # a quiet burst: hiss at least this many dB below normal...
QUIET_Z = 4.0              # ...and this many times the recording's usual wobble...
MIN_BURST_SEC = 4          # ...for at least this many seconds in a row
MIN_BURSTS = 3             # need this many separate bursts (1-2 could be a fluke)
BASELINE_SEC = 121         # window (s) for the normal hiss level
SILENCE_DB = 20.0          # a drop deeper than this is lost audio, not a signal

DROPOUT_FRACTION_MAX = 0.1 # >10% of frames with no audio at all -> flag
MIN_DURATION_S = 60.0      # recordings shorter than this are flagged

NEEDED_METADATA = ("station_lat", "station_lng", "station_alt", "tle1", "tle2")


def load_audio(path):
    """Read an audio file as mono float32. Returns (samples, sample_rate)."""
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), fs


def spectrogram_frames(x, fs):
    """Power spectrogram averaged into ~FRAME_SEC columns. Returns (freqs, times, S)."""
    nperseg = int(2 ** round(np.log2(fs / FREQ_RES_HZ)))
    freqs, times, S = signal.spectrogram(x, fs, window="hann", nperseg=nperseg,
                                         noverlap=nperseg // 2, detrend=False)
    hop = times[1] - times[0] if len(times) > 1 else nperseg / fs
    k = max(1, int(round(FRAME_SEC / hop)))
    n = S.shape[1] // k
    S = S[:, :n * k].reshape(S.shape[0], n, k).mean(axis=2)
    times = times[:n * k].reshape(n, k).mean(axis=1)
    return freqs, times, S


def usable_band(freqs, S):
    """
    Find where the recording actually has audio. Stations low-pass their audio at
    different cut-offs (5 kHz, 11 kHz, ...); above the cut-off there is only silence,
    which would confuse the noise estimate. Returns a boolean mask over freqs.
    """
    mean_db = 10 * np.log10(S.mean(axis=1) + 1e-30)
    ref = np.median(mean_db[(freqs > 200) & (freqs < 2000)])
    below = np.where((freqs > 1000) & (mean_db < ref - 10))[0]
    f_max = freqs[below[0]] - 300 if len(below) else freqs[-1] - 300
    return (freqs >= F_MIN_HZ) & (freqs <= f_max)


def snr_zscores(freqs, S):
    """
    Whiten the spectrogram: divide every frame by a running median across frequency
    (the local noise floor), then express each cell in robust standard deviations
    above that floor. Noise-only cells come out near 0; a real signal is large.
    """
    df = freqs[1] - freqs[0]
    win = int(NOISE_WINDOW_HZ / df) | 1
    floor = ndimage.median_filter(S, size=(win, 1), mode="nearest")
    ratio = S / (floor + 1e-30)
    center = np.median(ratio)
    spread = 1.4826 * np.median(np.abs(ratio - center))   # robust std (MAD)
    return (ratio - center) / (spread + 1e-12), ratio


def frame_peaks(freqs, z, ratio):
    """For each frame, the strongest peaks above Z_HIT: list of (frame, freq, z, ratio)."""
    df = freqs[1] - freqs[0]
    spacing = max(1, int(PEAK_SPACING_HZ / df))
    hits = []
    for j in range(z.shape[1]):
        col = z[:, j]
        idx, _ = signal.find_peaks(col, height=Z_HIT, distance=spacing)
        for i in idx[np.argsort(col[idx])[::-1][:PEAKS_PER_FRAME]]:
            hits.append((j, freqs[i], col[i], ratio[i, j]))
    return hits


def link_tracks(hits, frame_dt):
    """Chain peaks in consecutive frames into tracks that drift slowly in frequency."""
    tracks = []                                  # each: list of (frame, freq, z, ratio)
    for hit in sorted(hits):
        frame, freq = hit[0], hit[1]
        best, best_err = None, None
        for tr in tracks:
            last_frame, last_freq = tr[-1][0], tr[-1][1]
            gap = frame - last_frame
            if gap < 1 or gap > MAX_GAP_FRAMES + 1:
                continue
            allowed = MAX_DRIFT_HZ_PER_S * gap * frame_dt + FREQ_TOL_HZ
            err = abs(freq - last_freq)
            if err <= allowed and (best is None or err < best_err):
                best, best_err = tr, err
        if best is None:
            tracks.append([hit])
        else:
            best.append(hit)
    return tracks


def describe_track(track, times):
    """Summary numbers for one track."""
    frames = np.array([h[0] for h in track])
    f = np.array([h[1] for h in track])
    z = np.array([h[2] for h in track])
    snr_db = 10 * np.log10(np.array([h[3] for h in track]))
    t = times[frames]
    span = frames[-1] - frames[0] + 1
    slope = np.polyfit(t, f, 1)[0] if len(track) > 2 else 0.0
    # Residual Doppler bends gently, so judge smoothness against a parabola.
    smooth = np.polyval(np.polyfit(t, f, 2), t) if len(track) > 3 else f.mean()
    return {
        "hits": len(track),
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "fill": len(track) / span,
        "median_z": float(np.median(z)),
        "median_snr_db": float(np.median(snr_db)),
        "freq_start_hz": float(f[0]),
        "freq_end_hz": float(f[-1]),
        "drift_hz_per_s": float(slope),
        "scatter_hz": float(np.std(f - smooth)),
        "points": [(float(times[h[0]]), float(h[1])) for h in track],
    }


def find_wide_signal(freqs, S):
    """
    Look for a band ~BAND_WIDTH_HZ wide that stays at the same audio frequency and is
    louder than the spectrum around it for much of the pass. Returns a dict, or None.
    """
    m = max(1, int(round(BAND_BIN_HZ / (freqs[1] - freqs[0]))))
    n = len(freqs) // m
    f = freqs[:n * m].reshape(n, m).mean(axis=1)
    db = 10 * np.log10(S[:n * m].reshape(n, m, -1).mean(axis=1) + 1e-30)
    df = f[1] - f[0]

    band_db = ndimage.uniform_filter1d(db, max(1, int(BAND_WIDTH_HZ / df)), axis=0, mode="nearest")
    background = ndimage.median_filter(band_db, size=(int(BAND_BG_HZ / df) | 1, 1), mode="nearest")
    excess = band_db - background                       # dB louder than the surroundings
    # The averaging is unreliable at the edges of the audio, so leave them out.
    edge = int(BAND_WIDTH_HZ / df)
    if n <= 4 * edge:
        return None
    excess = excess[edge:-edge]
    f = f[edge:-edge]

    spread = 1.4826 * np.median(np.abs(excess - np.median(excess)))
    z = (excess - np.median(excess)) / (spread + 1e-12)
    best = np.argmax(z, axis=0)
    frames = np.arange(z.shape[1])
    strong = z[best, frames] >= BAND_Z_HIT
    if strong.sum() < 3:
        return None

    # The band's usual frequency: the most common position of the strongest band.
    hist, edges = np.histogram(f[best[strong]], bins=np.arange(f[0], f[-1] + 200, 200))
    centre = edges[np.argmax(hist)] + 100
    on = strong & (np.abs(f[best] - centre) <= BAND_FREQ_TOL_HZ)
    if not on.any():
        return None
    return {
        "freq_hz": float(np.median(f[best[on]])),
        "fill": float(on.mean()),
        "seconds": int(on.sum()),
        "excess_db": float(np.median(excess[best[on], frames[on]])),
    }


def analyse_audio(path):
    """Run the DSP detector on one recording. Returns a dict of measurements."""
    x, fs = load_audio(path)
    duration = len(x) / fs
    freqs, times, S = spectrogram_frames(x, fs)

    # Dropouts: frames whose total power collapses (lost samples / silence).
    power = S.sum(axis=0)
    dropout_frac = float(np.mean(power < np.median(power) / 10 ** (SILENCE_DB / 10)))

    band = usable_band(freqs, S)
    level_db, bursts = quiet_bursts(S[band], times)
    z, ratio = snr_zscores(freqs[band], S[band])
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else FRAME_SEC
    tracks = link_tracks(frame_peaks(freqs[band], z, ratio), frame_dt)
    tracks = [describe_track(t, times) for t in tracks if len(t) >= 3]
    tracks.sort(key=lambda d: (d["hits"], d["median_z"]), reverse=True)

    return {
        "duration_s": duration,
        "sample_rate": fs,
        "band_hz": (float(freqs[band][0]), float(freqs[band][-1])),
        "dropout_fraction": dropout_frac,
        "best_track": tracks[0] if tracks else None,
        "wide_signal": find_wide_signal(freqs[band], S[band]),
        "bursts": bursts,
        "_plot": (freqs[band], times, z, level_db),
    }


def quiet_bursts(S, times):
    """
    Find stretches where the total hiss level drops (the FM "quieting" a strong
    signal causes). Returns (level_db per frame relative to normal, list of bursts);
    each burst is {"start_s", "end_s", "depth_db"}.
    """
    level = 10 * np.log10(S.sum(axis=0) + 1e-30)
    # "Normal" hiss level: a 2-minute running median, so slow drifts in a station's
    # gain over the pass aren't mistaken for bursts.
    level = level - ndimage.median_filter(level, size=BASELINE_SEC, mode="nearest")
    smooth = ndimage.median_filter(level, size=3, mode="nearest")   # ignore 1-s blips
    # Second-to-second wobble (from differences, so bursts themselves don't inflate it).
    step = np.diff(smooth)
    wobble = 1.4826 * np.median(np.abs(step - np.median(step))) / np.sqrt(2)
    threshold = -max(QUIET_MIN_DB, QUIET_Z * wobble)
    quiet = (smooth < threshold) & (smooth > -SILENCE_DB)

    bursts, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= MIN_BURST_SEC:
                bursts.append({"start_s": float(times[start]), "end_s": float(times[i - 1]),
                               "depth_db": float(smooth[start:i].min())})
            start = None
    return level, bursts


def decide(measure, metadata, cnn_prob=None):
    """
    Turn measurements into a verdict. Returns (verdict, usable, reasons).

    cnn_prob is the waterfall CNN's probability (waterfall_cnn.py), or None if there
    is no waterfall image or no PyTorch. It is only a first-pass screen: a low score
    rejects the pass, a high score just lets the audio check decide.
    """
    if cnn_prob is not None and cnn_prob < CNN_THRESHOLD:
        return "NO_SIGNAL", False, [f"waterfall CNN screen says no signal "
                                    f"(score {cnn_prob:.2f} < {CNN_THRESHOLD:.2f})"]

    reasons, doubts = [], []
    tr = measure["best_track"]
    wide = measure.get("wide_signal")
    bursts = measure["bursts"]
    line_found = (tr is not None and tr["hits"] >= MIN_TRACK_HITS
                  and tr["fill"] >= MIN_TRACK_FILL and tr["median_z"] >= MIN_MEDIAN_Z)
    wide_found = (wide is not None and wide["fill"] >= BAND_MIN_FILL
                  and wide["excess_db"] >= BAND_MIN_EXCESS_DB)
    bursts_found = len(bursts) >= MIN_BURSTS

    if wide_found:
        # A steady wide band is what Doppler-corrected FUNcube-1 telemetry looks like.
        # Thin lines elsewhere in the same recording are usually other satellites.
        reasons.append(f"steady wide signal at {wide['freq_hz']:.0f} Hz in {wide['fill']:.0%} "
                       f"of the pass ({wide['excess_db']:.1f} dB above the noise)")
    if line_found and not wide_found:
        reasons.append(f"track {tr['freq_start_hz']:.0f}->{tr['freq_end_hz']:.0f} Hz for "
                       f"{tr['hits']} s (drift {tr['drift_hz_per_s']:+.2f} Hz/s, "
                       f"SNR {tr['median_snr_db']:.1f} dB)")
        if not bursts_found:         # the line is the only evidence, so its doubts count
            if abs(tr["drift_hz_per_s"]) > MAX_ACCEPT_DRIFT:
                doubts.append(f"drifts {tr['drift_hz_per_s']:+.1f} Hz/s, too fast for the "
                              "Doppler-corrected target (maybe another satellite)")
            if tr["scatter_hz"] > MAX_SCATTER_HZ:
                doubts.append(f"track is jumpy ({tr['scatter_hz']:.0f} Hz scatter)")
            if abs(tr["drift_hz_per_s"]) < SPUR_MAX_DRIFT:
                doubts.append("frequency never changes, so it can't be told apart "
                              "from local interference")
    if bursts_found:
        times = ", ".join(f"{b['start_s']:.0f}-{b['end_s']:.0f}" for b in bursts)
        reasons.append(f"{len(bursts)} quiet bursts where the hiss drops (at {times} s)")

    if not (wide_found or line_found or bursts_found):
        if (wide is not None and wide["fill"] >= BAND_WEAK_FILL
                and wide["excess_db"] >= BAND_WEAK_EXCESS_DB):
            reasons.append(f"weak wide signal at {wide['freq_hz']:.0f} Hz in {wide['fill']:.0%} "
                           f"of the pass ({wide['excess_db']:.1f} dB above the noise)")
            doubts.append("too weak or patchy to measure reliably")
        else:
            if tr is None:
                reasons.append("no narrowband signal above the noise")
            else:
                reasons.append(f"strongest track too weak/short ({tr['hits']} s of detections, "
                               f"fill {tr['fill']:.0%}, SNR {tr['median_snr_db']:.1f} dB)")
            reasons.append(f"{len(bursts)} quiet burst(s), need {MIN_BURSTS}")
            return "NO_SIGNAL", False, reasons

    if measure["dropout_fraction"] > DROPOUT_FRACTION_MAX:
        doubts.append(f"audio dropouts in {measure['dropout_fraction']:.0%} of the recording")
    if measure["duration_s"] < MIN_DURATION_S:
        doubts.append(f"recording only {measure['duration_s']:.0f} s long")
    if metadata and metadata.get("waterfall_status") == "without-signal":
        doubts.append("SatNOGS waterfall was vetted 'without-signal'")

    if doubts:
        return "UNCERTAIN", False, reasons + doubts
    return "SIGNAL", True, reasons


def metadata_status(metadata):
    """Which of the orbit-analysis essentials (station location, TLE) are missing."""
    if metadata is None:
        return ["no metadata file (re-run download_funcube.py)"]
    return [name for name in NEEDED_METADATA if metadata.get(name) in (None, "")]


def save_plot(path, measure, verdict, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs, times, z, level_db = measure["_plot"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.imshow(np.clip(z, -2, 12), aspect="auto", origin="lower", cmap="viridis",
              extent=[times[0], times[-1], freqs[0], freqs[-1]], interpolation="nearest")
    tr = measure["best_track"]
    if tr:
        pts = np.array(tr["points"])
        ax.plot(pts[:, 0], pts[:, 1], "r.", ms=3, label="best thin track")
    wide = measure.get("wide_signal")
    if wide:
        ax.axhline(wide["freq_hz"], color="orange", lw=1, ls="--", label="wide signal centre")
    if tr or wide:
        ax.legend(loc="upper right")
    ax.set_ylabel("audio frequency (Hz)")
    ax.set_title(f"{path.name}: {verdict}")

    ax2.plot(times, level_db, lw=0.8)
    for b in measure["bursts"]:
        ax2.axvspan(b["start_s"], b["end_s"], color="red", alpha=0.3)
    ax2.set_ylim(max(-10, level_db.min() - 0.5), max(1, level_db.max() + 0.5))
    ax2.set_ylabel("hiss level (dB)")
    ax2.set_xlabel("time since start of recording (s)  (red = quiet burst)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def validate_folder(folder, plots=True, use_cnn=True):
    """Validate every obs_*.ogg in `folder`; write validation_report.csv. Returns the rows."""
    folder = Path(folder)
    audio_files = sorted(folder.glob("obs_*.ogg"))
    if not audio_files:
        print(f"No obs_*.ogg files in {folder}")
        return []
    plot_dir = folder / "validation_plots"
    if plots:
        plot_dir.mkdir(exist_ok=True)

    rows = []
    for path in audio_files:
        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None

        waterfall = path.with_name(f"{path.stem}_waterfall.png")
        cnn_prob = waterfall_cnn.predict(waterfall) if use_cnn else None
        measure = analyse_audio(path)
        verdict, usable, reasons = decide(measure, metadata, cnn_prob)
        missing = metadata_status(metadata)
        tr = measure["best_track"] or {}
        wide = measure["wide_signal"] or {}

        print(f"{path.name:22s} {verdict:10s} {'usable' if usable else 'REJECT':7s} "
              f"{'; '.join(reasons)}")
        if plots:
            save_plot(path, measure, verdict, plot_dir / f"{path.stem}.png")

        rows.append({
            "file": path.name,
            "verdict": verdict,
            "usable_signal": usable,
            "metadata_complete": not missing,
            "missing_metadata": ", ".join(missing),
            "reasons": "; ".join(reasons),
            "cnn_score": None if cnn_prob is None else round(cnn_prob, 3),
            "duration_s": round(measure["duration_s"], 1),
            "file_size_mb": round(path.stat().st_size / 1e6, 2),
            "sample_rate": measure["sample_rate"],
            "dropout_fraction": round(measure["dropout_fraction"], 3),
            "track_seconds": tr.get("hits", 0),
            "track_snr_db": round(tr.get("median_snr_db", 0.0), 1),
            "track_freq_start_hz": round(tr.get("freq_start_hz", 0.0), 1),
            "track_freq_end_hz": round(tr.get("freq_end_hz", 0.0), 1),
            "track_drift_hz_per_s": round(tr.get("drift_hz_per_s", 0.0), 3),
            "wide_freq_hz": round(wide.get("freq_hz", 0.0), 1),
            "wide_fill": round(wide.get("fill", 0.0), 2),
            "wide_excess_db": round(wide.get("excess_db", 0.0), 1),
            "quiet_bursts": len(measure["bursts"]),
            "burst_times_s": " ".join(f"{b['start_s']:.0f}-{b['end_s']:.0f}"
                                      for b in measure["bursts"]),
            "station_name": (metadata or {}).get("station_name"),
            "station_lat": (metadata or {}).get("station_lat"),
            "station_lng": (metadata or {}).get("station_lng"),
            "station_alt": (metadata or {}).get("station_alt"),
            "start": (metadata or {}).get("start"),
        })

    report_csv = folder / "validation_report.csv"
    with open(report_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    good = [r for r in rows if r["usable_signal"]]
    total_mb = sum(r["file_size_mb"] for r in rows)
    total_min = sum(r["duration_s"] for r in rows) / 60
    print(f"\n{len(good)} of {len(rows)} passes have a validated signal "
          f"({sum(r['verdict'] == 'UNCERTAIN' for r in rows)} uncertain, "
          f"{sum(r['verdict'] == 'NO_SIGNAL' for r in rows)} no signal).")
    print(f"Metadata complete for {sum(r['metadata_complete'] for r in rows)} of {len(rows)}.")
    print(f"Dataset: {total_mb:.1f} MB for {total_min:.1f} min of audio "
          f"(~{total_mb / max(total_min, 1e-9):.2f} MB per minute).")
    print(f"Report: {report_csv}" + (f"  |  plots: {plot_dir}/" if plots else ""))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with obs_<id>.ogg (+ .json) files")
    parser.add_argument("--no-plots", action="store_true", help="skip spectrogram images")
    parser.add_argument("--no-cnn", action="store_true", help="skip the waterfall CNN screen")
    args = parser.parse_args()
    validate_folder(args.folder, plots=not args.no_plots, use_cnn=not args.no_cnn)


if __name__ == "__main__":
    main()
