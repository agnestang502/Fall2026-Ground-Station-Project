"""
Track the satellite's frequency through each validated pass and measure the residual.

How SatNOGS makes the audio (generic/bpsk.grc in satnogs-flowgraphs): the station
predicts the satellite's Doppler shift from its TLE and removes it, then shifts the
result up by exactly IF_HZ = 12,000 Hz so it becomes audible sound. So in the audio,
"exactly where the station expected the satellite" is 12,000 Hz, and

    residual (Hz) = measured audio frequency - 12,000 Hz
                  = observed radio frequency - frequency the station expected.

A residual of 0 everywhere would mean the TLE, the station's location and its clock
and radio were all perfect. A constant offset is mostly the station's radio being
slightly off (or the satellite's transmitter not being exactly on its listed
frequency). A residual that changes during the pass, especially an S-shape around
the moment of closest approach, means the predicted Doppler shift was slightly
wrong: the TLE (where the satellite really is) or the station's position is off.

How the peak is found, one second at a time:
  1. move the audio down so 12,000 Hz becomes 0 Hz (complex "baseband"),
  2. for BPSK (FUNcube-1 telemetry) square the signal. BPSK flips its phase by 180
     degrees to send bits, which smears its power over ~2 kHz with no peak to
     find. Squaring undoes the flips and leaves one sharp tone at exactly twice the
     signal's offset. (For a plain carrier, CW/FM, this step is skipped.)
  3. take the spectrum, find the tallest peak near the expected place, refine it
     between frequency bins with a parabola fit, and halve it again for BPSK,
  4. throw away seconds where the peak is not clearly above the noise or jumps
     away from its neighbours.

Usage:
    python track_peaks.py data/39444            # every SIGNAL pass in the report
    python track_peaks.py data/39444 --all      # every pass, even rejected ones
Writes data/<norad>/peak_tracks/obs_<id>.csv, obs_<id>.png and residual_summary.csv.
"""
import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

import validate_signals as vs

IF_HZ = 12000.0           # SatNOGS shifts the Doppler-corrected signal up by this much
BASEBAND_RATE = 16000     # samples/s after moving 12 kHz to 0 Hz (keeps +-8 kHz)
FRAME_SEC = 1.0           # one frequency measurement per second
SEARCH_HZ = 3000.0        # look for the satellite within +-3 kHz of where it should be
CARRIER_SNR_DB = 15.0     # BPSK: a sharp line this strong before squaring is not BPSK
MIN_PEAK_SNR_DB = 10.0    # a second's peak must be this far above that spectrum's median
MAX_STEP_HZ = 20.0        # neighbouring good seconds may differ by at most this much
OUTLIER_MAD = 4.0         # drop points this many robust std-devs from the smooth track
MEDIAN_WINDOW = 15        # seconds, for the smooth track used to spot outliers
JOIN_DRIFT_HZ_PER_S = 3.0 # across a gap, the residual may have moved this fast
MIN_SEGMENT_POINTS = 3    # shorter runs of points away from the main track are dropped


def to_baseband(x, fs):
    """Move IF_HZ down to 0 Hz and resample to BASEBAND_RATE. Returns complex samples."""
    t = np.arange(len(x)) / fs
    z = (x * np.exp(-2j * np.pi * IF_HZ * t)).astype(np.complex64)
    g = np.gcd(int(fs), BASEBAND_RATE)
    # resample_poly low-pass filters as it resamples, which removes the mirror image
    # that the shift creates near -24 kHz.
    return signal.resample_poly(z, BASEBAND_RATE // g, int(fs) // g)


def parabolic_peak(power, i):
    """Refine a spectrum peak at bin i to a fraction of a bin. Returns the offset in bins."""
    if i <= 0 or i >= len(power) - 1:
        return 0.0
    a, b, c = np.log(power[i - 1:i + 2] + 1e-30)
    denom = a - 2 * b + c
    return 0.0 if denom == 0 else 0.5 * (a - c) / denom


def frame_measurements(z, square, guess_hz):
    """
    One (time_s, offset_hz, snr_db) per FRAME_SEC of baseband z.
    guess_hz: where to centre the search (None = anywhere within SEARCH_HZ).

    For BPSK, a steady tone (interference, another transmitter's carrier) would also
    give a sharp peak after squaring. Real BPSK has no sharp peak *before* squaring,
    so frequencies where the unsquared spectrum already has one are skipped.
    """
    n = int(FRAME_SEC * BASEBAND_RATE)
    nfft = 4 * n                                     # zero-pad: finer bins for the fit
    factor = 2.0 if square else 1.0                  # squaring doubles the frequency
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / BASEBAND_RATE)) / factor
    centre = 0.0 if guess_hz is None else guess_hz
    window = np.hanning(n)
    keep = np.abs(freqs - centre) <= SEARCH_HZ
    idx = np.where(keep)[0]
    out = []
    raw_freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / BASEBAND_RATE))
    for k in range(len(z) // n):
        seg = z[k * n:(k + 1) * n]
        allowed = idx
        if square:
            raw = np.fft.fftshift(np.abs(np.fft.fft(seg * window, nfft)) ** 2)
            raw_snr = 10 * np.log10(raw / (ndimage.median_filter(raw, 801) + 1e-30))
            tone = ndimage.maximum_filter1d(raw_snr, 25) > CARRIER_SNR_DB   # +-3 Hz
            tone_at = np.interp(freqs[idx], raw_freqs, tone.astype(float)) > 0.5
            allowed = idx[~tone_at]
            if len(allowed) == 0:
                out.append(((k + 0.5) * FRAME_SEC, np.nan, -np.inf))
                continue
            seg = seg ** 2
        p = np.fft.fftshift(np.abs(np.fft.fft(seg * window, nfft)) ** 2)
        i = allowed[np.argmax(p[allowed])]
        snr_db = 10 * np.log10(p[i] / (np.median(p[idx]) + 1e-30))
        f = freqs[i] + parabolic_peak(p, i) * (freqs[1] - freqs[0])
        out.append(((k + 0.5) * FRAME_SEC, f, snr_db))
    return np.array(out).reshape(-1, 3)


def clean_track(m):
    """Keep seconds whose peak is strong and agrees with its neighbours. Returns a mask."""
    t, f, snr = m.T
    good = (snr >= MIN_PEAK_SNR_DB) & np.isfinite(f)
    if good.sum() < 5:
        return good
    # Smooth track through the strong points; drop anything far from it.
    smooth = np.interp(t, t[good], ndimage.median_filter(f[good], MEDIAN_WINDOW, mode="nearest"))
    dev = f - smooth
    spread = 1.4826 * np.median(np.abs(dev[good] - np.median(dev[good]))) + 0.5
    good &= np.abs(dev) <= max(OUTLIER_MAD * spread, 2.0)
    return good & main_track(t, f, good)


def main_track(t, f, good):
    """
    Keep only the satellite's own track. Good points are chained into segments that
    move smoothly (<= MAX_STEP_HZ between neighbours); the longest segment is the
    satellite, and other segments are kept only if they continue it plausibly
    (another signal crossing the band, or noise, would start somewhere else).
    """
    gi = np.where(good)[0]
    if len(gi) < 2:
        return good
    segments, current = [], [gi[0]]
    for a, b in zip(gi[:-1], gi[1:]):
        if abs(f[b] - f[a]) <= MAX_STEP_HZ + JOIN_DRIFT_HZ_PER_S * (t[b] - t[a] - FRAME_SEC):
            current.append(b)
        else:
            segments.append(current)
            current = [b]
    segments.append(current)
    segments.sort(key=len, reverse=True)

    kept = list(segments[0])
    for seg in segments[1:]:
        if len(seg) < MIN_SEGMENT_POINTS:
            continue
        # Compare the segment's nearest end with the nearest kept point in time.
        kt = t[kept]
        for end in (seg[0], seg[-1]):
            j = kept[int(np.argmin(np.abs(kt - t[end])))]
            gap = abs(t[end] - t[j])
            if abs(f[end] - f[j]) <= MAX_STEP_HZ + JOIN_DRIFT_HZ_PER_S * gap:
                kept.extend(seg)
                break
    mask = np.zeros_like(good)
    mask[kept] = True
    return mask


def track_pass(audio_path, metadata=None, guess_hz=None):
    """Measure the residual through one pass. Returns a dict with the track and a summary."""
    mode = (metadata or {}).get("transmitter_mode") or "BPSK"
    square = "PSK" in mode.upper()
    x, fs = vs.load_audio(audio_path)
    z = to_baseband(x, fs)
    m = frame_measurements(z, square, guess_hz)
    good = clean_track(m)
    t, f, snr = m[good].T if good.any() else (np.array([]),) * 3

    summary = {"mode": mode, "points": int(good.sum()), "seconds": len(m)}
    if len(t) >= 5:
        slope, _ = np.polyfit(t, f, 1)
        summary.update({
            "median_residual_hz": float(np.median(f)),
            "residual_change_hz": float(np.percentile(f, 95) - np.percentile(f, 5)),
            "residual_slope_hz_per_min": float(slope * 60),
            "track_scatter_hz": float(np.std(f - ndimage.median_filter(f, 9, mode="nearest"))),
            "median_peak_snr_db": float(np.median(snr)),
        })
    return {"all": m, "good": good, "summary": summary}


def save_track_csv(path, result, start_utc):
    m, good = result["all"], result["good"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_utc", "time_s", "audio_freq_hz", "residual_hz", "peak_snr_db", "used"])
        for (t, f, snr), g in zip(m, good):
            utc = (start_utc + timedelta(seconds=float(t))).isoformat() if start_utc else ""
            w.writerow([utc, f"{t:.1f}", f"{IF_HZ + f:.2f}", f"{f:.2f}", f"{snr:.1f}", int(g)])


def save_track_plot(path, audio_path, result, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, good = result["all"], result["good"]
    x, fs = vs.load_audio(audio_path)
    freqs, times, S = vs.spectrogram_frames(x, fs)
    near = np.abs(freqs - IF_HZ) <= SEARCH_HZ
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1]})
    db = 10 * np.log10(S[near] + 1e-30)
    ax.imshow(db, aspect="auto", origin="lower", cmap="viridis",
              extent=[times[0], times[-1], freqs[near][0] - IF_HZ, freqs[near][-1] - IF_HZ],
              vmin=np.percentile(db, 5), vmax=np.percentile(db, 99.5))
    ax.plot(m[good, 0], m[good, 1], "r.", ms=2, label="tracked peak")
    ax.axhline(0, color="w", lw=0.6, ls=":", label="where the station expected it")
    ax.set_ylabel("offset from expected (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)

    ax2.plot(m[~good, 0], m[~good, 1], ".", color="0.8", ms=2, label="rejected seconds")
    ax2.plot(m[good, 0], m[good, 1], "r.", ms=3, label="residual")
    ax2.axhline(0, color="k", lw=0.6, ls=":")
    if good.sum() >= 5:
        lo, hi = np.percentile(m[good, 1], [1, 99])
        pad = max(10.0, 0.3 * (hi - lo))
        ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_ylabel("residual (Hz)")
    ax2.set_xlabel("time since start of recording (s)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def read_report(folder):
    path = folder / "validation_report.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run validate_signals.py {folder} first")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def guess_from_report(row):
    """Start the search where the validator saw the wide signal, if it did."""
    f = float(row.get("wide_freq_hz") or 0)
    return f - IF_HZ if f and abs(f - IF_HZ) < SEARCH_HZ else None


def track_folder(folder, include_all=False, plots=True):
    folder = Path(folder)
    out = folder / "peak_tracks"
    out.mkdir(exist_ok=True)
    rows = []
    for row in read_report(folder):
        if not include_all and row["verdict"] != "SIGNAL":
            continue
        audio = folder / row["file"]
        meta_path = audio.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        start = datetime.fromisoformat(meta["start"].replace("Z", "+00:00")) if meta.get("start") else None
        result = track_pass(audio, meta, guess_from_report(row))
        s = result["summary"]
        save_track_csv(out / f"{audio.stem}.csv", result, start)
        if plots:
            title = f"{audio.stem}  {meta.get('station_name', '')}  max elevation {meta.get('max_altitude', '?')}°"
            save_track_plot(out / f"{audio.stem}.png", audio, result, title)
        rows.append({"file": row["file"], "verdict": row["verdict"],
                     "station_name": meta.get("station_name"), "start": meta.get("start"),
                     "max_altitude": meta.get("max_altitude"), **s})
        med = s.get("median_residual_hz")
        print(f"{audio.stem:16s} {s['points']:4d}/{s['seconds']:<4d} s tracked  " +
              (f"median residual {med:+8.1f} Hz, changes {s['residual_change_hz']:6.1f} Hz "
               f"over the pass ({meta.get('station_name', '')})" if med is not None
               else "too few clean points"))
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "residual_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nTracked {len(rows)} passes. Results in {out}/")
    else:
        print("No SIGNAL passes in the report (use --all to track every pass).")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with validation_report.csv")
    parser.add_argument("--all", action="store_true", help="also track passes not marked SIGNAL")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    track_folder(args.folder, include_all=args.all, plots=not args.no_plots)


if __name__ == "__main__":
    main()
