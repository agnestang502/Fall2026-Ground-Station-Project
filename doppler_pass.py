"""
Week 3 of the satellite-pass simulation: work in 3D and compute the Doppler shift.

Takes the highest pass over the station (from Week 2's predict_passes.py), gets the
satellite's and station's positions and velocities every second as 3D vectors in a
frame that doesn't rotate with Earth, and then:
  * draws Earth, one full orbit, the station and the line of sight in 3D (plotly),
  * computes the range-rate three ways (dot product, finite difference, skyfield's
    frame_latlon_and_rates) and plots them together,
  * turns range-rate into the received frequency, f_r = f_0 (1 - range_rate / c),
    and plots the Doppler shift f_r - f_0 against time.

Week 3 checkpoint:
  * the three range-rate curves agree to within about 1 m/s,
  * the Doppler curve crosses zero at the moment of minimum range, and
  * the peak shift is close to f_0 x 7.5 km/s / c (about 10.9 kHz at 437.6 MHz).

Usage:
    python doppler_pass.py                        # highest UmKA-1 pass over station 5049
    python doppler_pass.py --stretch              # also compare a high and a low pass
    python doppler_pass.py --tle-file my.tle --start 2026-11-02T01:01:22   # reproduce a run
    python doppler_pass.py --norad 59112 --freq-hz 145840000   # another satellite

Distances and speeds are converted to meters and m/s as soon as they leave skyfield.
"""
import argparse
from pathlib import Path

import numpy as np

import find_satellite as fs
import predict_passes as pp

C_M_S = 299_792_458.0            # speed of light, m/s
LEO_SPEED_M_S = 7_500.0          # the plan's typical low-Earth-orbit speed, for the checkpoint
EARTH_RADIUS_KM = 6371.0         # mean radius, for drawing the globe
STEP_S = 1                       # sampling step across the pass

AGREE_M_S = 1.0                  # Week 3 checkpoint: range-rate methods agree to this
ZERO_CROSS_S = 1.0               # Doppler zero crossing within this of minimum range
PEAK_SHIFT_RANGE = (0.75, 1.05)  # peak |shift| as a fraction of f_0 x 7.5 km/s / c

PLOT_DIR = pp.PLOT_DIR


# --- Vectors --------------------------------------------------------------------

def times_after(t0, seconds):
    """skyfield Times `seconds` after t0.

    The date is passed as a whole day plus a fraction: one float holding a Julian
    date like 2461352.5 is only good to about 40 microseconds, which is 30 cm of
    satellite motion and shows up as a sawtooth in the finite-difference range-rate."""
    return fs.ts.tt_jd(t0.whole, t0.tt_fraction + np.asarray(seconds) / 86400)


def pass_times(p, step_s=STEP_S):
    """Seconds since rise, and the matching skyfield Times, from rise to set."""
    seconds = np.arange(0, (p["set"] - p["rise"]) * 86400 + step_s / 2, step_s)
    return seconds, times_after(p["rise"], seconds)


def state_vectors(sat, station, t):
    """Positions (m) and velocities (m/s) of the satellite and the station, each a
    3 x N array in the GCRS frame: centred on Earth, not turning with it."""
    s, g = sat.at(t), station.at(t)
    return (s.position.km * 1e3, s.velocity.km_per_s * 1e3,
            g.position.km * 1e3, g.velocity.km_per_s * 1e3)


def range_rate_dot(r_sat, v_sat, r_sta, v_sta):
    """Range (m) and range-rate (m/s) by hand: rho_dot = (rho / |rho|) . (v_sat - v_station).

    Positive range-rate means the satellite is moving away from the station."""
    rho = r_sat - r_sta                              # station -> satellite, 3 x N
    rng = np.linalg.norm(rho, axis=0)
    rate = np.sum((rho / rng) * (v_sat - v_sta), axis=0)
    return rng, rate


def range_rate_finite_difference(rng, step_s=STEP_S):
    """Range-rate (m/s) from the change in range between samples.

    np.gradient takes the centred difference (r[i+1] - r[i-1]) / 2 dt inside the pass,
    and a second-order one-sided difference at the two ends."""
    return np.gradient(rng, step_s, edge_order=2)


def range_rate_skyfield(sat, station, t):
    """Range-rate (m/s) from skyfield: the sixth value of frame_latlon_and_rates."""
    return (sat - station).at(t).frame_latlon_and_rates(station)[5].km_per_s * 1e3


def doppler_hz(f0, rate):
    """Received frequency f_r = f_0 (1 - range_rate / c) and the shift f_r - f_0 (Hz)."""
    f_r = f0 * (1 - rate / C_M_S)
    return f_r, f_r - f0


# --- Checks -------------------------------------------------------------------------

def zero_crossing_s(seconds, values):
    """Time (s) where values first changes sign, by linear interpolation, or None."""
    sign = np.sign(values)
    idx = np.nonzero(sign[:-1] * sign[1:] < 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    return seconds[i] - values[i] * (seconds[i + 1] - seconds[i]) / (values[i + 1] - values[i])


def min_range_s(seconds, rng):
    """Time (s) of minimum range, refined with a parabola through the lowest 3 samples."""
    i = int(np.clip(np.argmin(rng), 1, len(rng) - 2))
    y0, y1, y2 = rng[i - 1:i + 2]
    step = seconds[i + 1] - seconds[i]
    return seconds[i] + 0.5 * step * (y0 - y2) / (y0 - 2 * y1 + y2)


def reference_shift_hz(f0):
    """The plan's yardstick for the peak shift: f_0 x 7.5 km/s / c."""
    return f0 * LEO_SPEED_M_S / C_M_S


def checkpoint(seconds, rng, rates, shift, f0):
    """Return (list of failed checks, dict of measured values); no failures = PASS."""
    dot = rates["dot product"]
    worst = {name: float(np.max(np.abs(r - dot))) for name, r in rates.items() if name != "dot product"}
    t_zero, t_min = zero_crossing_s(seconds, shift), min_range_s(seconds, rng)
    peak = float(np.max(np.abs(shift)))
    ratio = peak / reference_shift_hz(f0)
    found = {"worst_diff_m_s": worst, "zero_cross_s": t_zero, "min_range_s": t_min,
             "peak_shift_hz": peak, "peak_ratio": ratio}

    problems = []
    for name, diff in worst.items():
        if diff > AGREE_M_S:
            problems.append(f"{name} differs from the dot product by {diff:.2f} m/s")
    if t_zero is None:
        problems.append("the Doppler shift never crosses zero")
    elif abs(t_zero - t_min) > ZERO_CROSS_S:
        problems.append(f"zero crossing at {t_zero:.1f} s but minimum range at {t_min:.1f} s")
    if not (shift[0] > 0 > shift[-1]):
        problems.append("shift should start positive (approaching) and end negative")
    lo, hi = PEAK_SHIFT_RANGE
    if not lo <= ratio <= hi:
        problems.append(f"peak shift {peak:.0f} Hz is {ratio:.2f} x f_0 x 7.5 km/s / c")
    return problems, found


# --- Plots ----------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_range_rate(seconds, rates, p, title, out_path):
    """The three range-rate curves on top of each other, and their differences below."""
    plt = _plt()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    minutes = seconds / 60
    styles = {"dot product": dict(lw=4, alpha=0.35), "finite difference": dict(lw=2, ls="--"),
              "skyfield": dict(lw=1.2, ls=":", color="k")}
    for name, rate in rates.items():
        top.plot(minutes, rate / 1e3, label=name, **styles.get(name, {}))
    top.axhline(0, color="gray", lw=0.8)
    top.set_ylabel("Range-rate (km/s)\n+ = moving away")
    top.legend(loc="upper left")
    top.grid(alpha=0.3)
    top.set_title(title)

    dot = rates["dot product"]
    for name, rate in rates.items():
        if name != "dot product":
            bottom.plot(minutes, (rate - dot) * 1e3, label=f"{name} − dot product")
    bottom.axhline(0, color="gray", lw=0.8)
    bottom.set_ylabel("Difference (mm/s)")
    bottom.set_xlabel(f"Minutes after rise ({pp.utc(p['rise'])} UTC)")
    bottom.legend(loc="upper left")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_doppler(seconds, shift, rng, found, f0, p, title, out_path):
    """Doppler shift against time, with minimum range and the zero crossing marked."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    minutes = seconds / 60
    ax.plot(minutes, shift / 1e3, lw=2, label=r"$f_r - f_0$")
    ref = reference_shift_hz(f0) / 1e3
    for y in (ref, -ref):
        ax.axhline(y, color="gray", ls="--", lw=0.8)
    ax.text(minutes[-1], ref, f" ±{ref:.1f} kHz\n (f₀ × 7.5 km/s ÷ c)", va="center", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(found["min_range_s"] / 60, color="tab:red", ls=":",
               label=f"minimum range {rng.min() / 1e3:.0f} km")
    ax.plot(found["zero_cross_s"] / 60, 0, "o", color="tab:red", label="zero crossing")
    ax.set_xlabel(f"Minutes after rise ({pp.utc(p['rise'])} UTC)")
    ax.set_ylabel(f"Doppler shift (kHz) around f₀ = {f0 / 1e6:.3f} MHz")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_3d(sat, station, p, title, out_path, n_lines=5):
    """Interactive 3D view (plotly HTML): Earth, one full orbit centred on the pass,
    the pass itself, the station and the line of sight at a few moments."""
    import plotly.graph_objects as go

    # One full orbit, centred on the moment the pass culminates.
    period_s = fs.period_minutes(sat) * 60
    orbit_s = np.linspace(-period_s / 2, period_s / 2, 721)
    orbit = sat.at(times_after(p["culminate"], orbit_s)).position.km
    seconds, t = pass_times(p)
    r_sat = sat.at(t).position.km
    r_sta = station.at(t).position.km

    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    earth = go.Surface(x=EARTH_RADIUS_KM * np.cos(u) * np.sin(v),
                       y=EARTH_RADIUS_KM * np.sin(u) * np.sin(v),
                       z=EARTH_RADIUS_KM * np.cos(v),
                       colorscale=[[0, "#1f4e79"], [1, "#2e86c1"]], showscale=False,
                       opacity=0.9, name="Earth", hoverinfo="skip")
    traces = [
        earth,
        go.Scatter3d(x=orbit[0], y=orbit[1], z=orbit[2], mode="lines", name="one full orbit",
                     line=dict(color="lightgray", width=3)),
        go.Scatter3d(x=r_sat[0], y=r_sat[1], z=r_sat[2], mode="lines", name="the pass",
                     line=dict(color="orange", width=7)),
        go.Scatter3d(x=r_sta[0, :1], y=r_sta[1, :1], z=r_sta[2, :1], mode="markers",
                     name="station", marker=dict(color="red", size=6)),
    ]
    for k, i in enumerate(np.linspace(0, len(seconds) - 1, n_lines).round().astype(int)):
        traces.append(go.Scatter3d(
            x=[r_sta[0, i], r_sat[0, i]], y=[r_sta[1, i], r_sat[1, i]], z=[r_sta[2, i], r_sat[2, i]],
            mode="lines+markers", line=dict(color="yellow", width=4), marker=dict(size=3),
            name="line of sight", legendgroup="los", showlegend=k == 0,
            hovertext=f"{seconds[i] / 60:.1f} min after rise"))
    fig = go.Figure(traces)
    # Look from beside the orbit plane, tilted toward the station, so the orbit shows
    # as a ring and the pass and lines of sight sit in front of the globe.
    mid = len(seconds) // 2
    normal = np.cross(r_sat[:, mid], sat.at(t[mid]).velocity.km_per_s)
    eye = 0.75 * normal / np.linalg.norm(normal) + 0.65 * r_sta[:, mid] / np.linalg.norm(r_sta[:, mid])
    eye = 1.9 * eye / np.linalg.norm(eye)
    axis = dict(showbackground=False, gridcolor="#333", zerolinecolor="#555", color="#aaa")
    fig.update_layout(title=title, paper_bgcolor="black", font_color="white",
                      scene=dict(aspectmode="data", bgcolor="black",
                                 xaxis=dict(axis, title="x (km)"), yaxis=dict(axis, title="y (km)"),
                                 zaxis=dict(axis, title="z (km)"),
                                 camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
                      legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True)
    return out_path


def analyse_pass(sat, station, p, f0):
    """Everything Week 3 computes for one pass, as a dict of arrays."""
    return analyse_times(sat, station, *pass_times(p), f0)


def analyse_times(sat, station, seconds, t, f0):
    """Range, the three range-rates and the Doppler shift at times t (seconds apart)."""
    rng, dot = range_rate_dot(*state_vectors(sat, station, t))
    rates = {"dot product": dot,
             "finite difference": range_rate_finite_difference(rng),
             "skyfield": range_rate_skyfield(sat, station, t)}
    f_r, shift = doppler_hz(f0, dot)
    return {"seconds": seconds, "range_m": rng, "rates": rates, "f_r": f_r, "shift": shift}


def plot_high_low(sat, station, high, low, f0, title, out_path, half_window_s=300):
    """Stretch: Doppler curves of a high and a low pass, over the same window around
    closest approach. Solid where the satellite is above 10 deg, dotted below."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    seconds = np.arange(-half_window_s, half_window_s + STEP_S, STEP_S)
    for p, label, color in ((high, "high", "tab:blue"), (low, "low", "tab:orange")):
        t = times_after(p["culminate"], seconds)
        a = analyse_times(sat, station, seconds, t, f0)
        t_close = min_range_s(seconds, a["range_m"])
        el = pp.elevation_deg(sat, station, t)
        slope = np.gradient(a["shift"], seconds)
        steepest = float(np.min(slope))                  # Hz/s, at closest approach
        x = (seconds - t_close) / 60
        up = np.where(el >= pp.MIN_ELEVATION_DEG, a["shift"], np.nan)
        ax.plot(x, a["shift"] / 1e3, ls=":", color=color, lw=1.2)
        ax.plot(x, up / 1e3, color=color, lw=2.5,
                label=f"{label} pass: peak {p['peak_deg']:.0f}°, closest {a['range_m'].min() / 1e3:.0f} km, "
                      f"steepest {steepest:.0f} Hz/s")
        in_pass = el >= pp.MIN_ELEVATION_DEG
        rows.append((label, p, a["range_m"].min(), steepest,
                     float(np.max(np.abs(a["shift"][in_pass])))))
    ax.plot([], [], ls=":", color="gray", label=f"below {pp.MIN_ELEVATION_DEG}° (not counted in the pass)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Minutes from closest approach")
    ax.set_ylabel(f"Doppler shift (kHz) around f₀ = {f0 / 1e6:.3f} MHz")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, rows


def main():
    p = argparse.ArgumentParser(description="Week 3: 3D view, range-rate and Doppler shift.")
    pp.add_pass_args(p)
    p.add_argument("--freq-hz", type=float, default=fs.BEACON_HZ, help="beacon frequency f_0 (Hz)")
    p.add_argument("--stretch", action="store_true",
                   help="also plot the Doppler curves of the highest and lowest passes together")
    args = p.parse_args()

    sat, record, station, t0 = pp.load_inputs(args)
    f0 = args.freq_hz
    passes = pp.find_passes(sat, station, t0, t0 + args.days)
    if not passes:
        raise SystemExit("No passes above the minimum elevation in this window")
    best = max(passes, key=lambda q: q["peak_deg"])
    print(f"Beacon f_0     : {f0 / 1e6:.3f} MHz")
    print(f"Chosen pass    : highest of {len(passes)}, rise {pp.utc(best['rise'])} UTC, "
          f"peak {best['peak_deg']:.1f}°, {best['duration_min']:.1f} min")

    a = analyse_pass(sat, station, best, f0)
    seconds, rng, rates, shift = a["seconds"], a["range_m"], a["rates"], a["shift"]
    problems, found = checkpoint(seconds, rng, rates, shift, f0)
    rise_offset = best["rise"].utc_datetime()

    def at(s):
        return f"{s:6.1f} s after rise ({(rise_offset.timestamp() + s) % 86400 // 3600:02.0f}:" \
               f"{(rise_offset.timestamp() + s) % 3600 // 60:02.0f}:{(rise_offset.timestamp() + s) % 60:04.1f} UTC)"

    print(f"Range          : {rng.max() / 1e3:.0f} km at rise -> {rng.min() / 1e3:.0f} km closest")
    print(f"Range-rate     : {rates['dot product'][0] / 1e3:+.3f} km/s at rise -> "
          f"{rates['dot product'][-1] / 1e3:+.3f} km/s at set")
    for name, diff in found["worst_diff_m_s"].items():
        print(f"  {name:<17}: largest difference from dot product {diff * 1e3:.2f} mm/s")
    print(f"Doppler shift  : {shift[0]:+.0f} Hz at rise -> {shift[-1]:+.0f} Hz at set")
    print(f"Peak |shift|   : {found['peak_shift_hz']:.0f} Hz = {found['peak_ratio']:.2f} x "
          f"f_0 x 7.5 km/s / c ({reference_shift_hz(f0):.0f} Hz)")
    print(f"Minimum range  : {at(found['min_range_s'])}")
    print(f"Zero crossing  : {at(found['zero_cross_s'])}" if found["zero_cross_s"] is not None
          else "Zero crossing  : none")
    print("Checkpoint     : " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    tag = f"{sat.model.satnum}_{record['id'] or 'custom'}"
    where = record["name"] + (f" (SatNOGS station {record['id']})" if record["id"] else "")
    day = best["rise"].utc_strftime("%Y-%m-%d")
    print(f"Range-rate plot: {plot_range_rate(seconds, rates, best, f'{sat.name} range-rate over {where}, {day}', PLOT_DIR / f'range_rate_{tag}.png')}")
    print(f"Doppler plot   : {plot_doppler(seconds, shift, rng, found, f0, best, f'{sat.name} Doppler shift over {where}, {day}', PLOT_DIR / f'doppler_{tag}.png')}")
    print(f"3D view        : {plot_3d(sat, station, best, f'{sat.name} pass over {where}, {day} (GCRS frame)', PLOT_DIR / f'pass3d_{tag}.html')}")

    if args.stretch:
        low = min(passes, key=lambda q: q["peak_deg"])
        out, rows = plot_high_low(sat, station, best, low, f0,
                                  f"{sat.name}: a high and a low pass over {where}",
                                  PLOT_DIR / f"doppler_high_low_{tag}.png")
        print("\nStretch: high pass vs low pass")
        for label, q, closest, steepest, peak in rows:
            print(f"  {label:<4} pass: rise {pp.utc(q['rise'])} UTC, peak {q['peak_deg']:.1f}°, "
                  f"closest {closest / 1e3:.0f} km, peak |shift| {peak:.0f} Hz, "
                  f"steepest change {steepest:.0f} Hz/s")
        print(f"  Plot: {out}")


if __name__ == "__main__":
    main()
