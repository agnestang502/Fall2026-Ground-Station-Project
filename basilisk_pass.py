"""
Propagate the satellite with Basilisk and watch the pass in Vizard.

Basilisk (CU Boulder's Autonomous Vehicle Systems lab) is a spacecraft simulation
framework, and Vizard is its 3D viewer. This script:
  * hands the TLE to Basilisk (Basilisk's own tleHandling turns it into a position and
    velocity at the TLE's epoch),
  * propagates the orbit with Basilisk's integrator and a 10 x 10 GGM03S Earth gravity
    field, plus the Sun's pull,
  * puts the SatNOGS station on the rotating Earth with Basilisk's groundLocation
    module, which reports elevation, range and range-rate every second,
  * compares the result with skyfield/SGP4 (Weeks 2 and 3): how far apart the two
    orbits drift, the pass times, and the Doppler curve, and
  * saves the pass for Vizard, or streams it to Vizard live in real time (--live).

In Vizard you see the Earth turning, the orbit, the station with its 10 deg elevation
cone, a line from the station to the satellite while it can hear it, the beacon
lighting up during the pass, and a camera on the satellite looking straight down.

Usage:
    python basilisk_pass.py                      # highest pass in the next day, saved for Vizard
    python basilisk_pass.py --days 7             # highest pass in the next 7 days
    python basilisk_pass.py --tle-file my.tle --start 2026-11-08T23:15:17   # reproduce a run
    python basilisk_pass.py --live               # stream to Vizard in real time from now
    python basilisk_pass.py --live --speed 20    # ... 20 times faster than real time
    python basilisk_pass.py --live --no-viz      # real-time status lines only, no Vizard

The station and window options are the same as Week 2's predict_passes.py. Basilisk is
installed with `pip install bsk`. Vizard is a separate download (see docs/basilisk_vizard.md).

Basilisk works in meters, seconds and radians throughout. Simulation time 0 is the
TLE's epoch.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from skyfield.framelib import itrs

import doppler_pass as dp
import find_satellite as fs
import predict_passes as pp

try:
    from Basilisk.architecture import messaging, sysModel
    from Basilisk.simulation import groundLocation, simSynch, spacecraft, vizInterface
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, tleHandling, vizSupport
    from Basilisk.utilities import orbitalMotion as om
    from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path
except ImportError as err:
    raise SystemExit(f"Basilisk is not installed ({err}). Install it with: pip install bsk")

STEP_S = 1                       # integration and sampling step
GRAV_DEGREE = 10                 # degree and order of the GGM03S gravity field
OMEGA_EARTH = 7.2921158553e-5    # [rad/s] Earth's rotation rate relative to the stars
EARTH_RADIUS_M = 6378136.3       # GGM03S reference radius, also Basilisk's
AU_M = 149_597_870_700.0
DAYS_AHEAD = 1                   # default window: Basilisk drifts from SGP4 over days (no drag)
VIZ_MARGIN_S = 300               # save this much before rise and after set for Vizard
CHUNK_S = 3600                   # Earth orientation is computed this many seconds at a time
BEACON_BAUD = 1.0                # any negative baud rate makes Vizard draw "sending"

START_MATCH_M = 10.0             # checkpoint: Basilisk's starting state matches SGP4 to this
RANGE_RATE_MATCH_M_S = 0.01      # checkpoint: groundLocation range-rate vs the Week 3 formula

VIZ_DIR = Path(__file__).parent / "data" / "vizard"
PLOT_DIR = pp.PLOT_DIR


# --- The TLE, Earth's rotation and the Sun ----------------------------------------------

def tle_path(args):
    """The TLE file fetch_tle read or saved."""
    return Path(args.tle_file) if args.tle_file else fs.TLE_DIR / f"{args.norad}.tle"


def tle_state(path, norad):
    """Position (m) and velocity (m/s) in Basilisk's inertial frame at the TLE's epoch,
    from Basilisk's own TLE reader (SGP4 at the epoch, then TEME -> GCRF)."""
    entries = tleHandling.satTle2elem(str(path))
    entry = next((e for e in entries if int(e.noradID) == norad), None)
    if entry is None:
        raise SystemExit(f"No TLE for NORAD {norad} in {path}")
    # satTle2elem made the elements with this mu, so going back with it is exact.
    r, v = om.elem2rv(om.MU_EARTH * 1e9, entry.oe)
    return np.asarray(r, float), np.asarray(v, float)


def earth_rotation(t):
    """Rotation from the inertial frame (GCRS) to the Earth-fixed frame (ITRS) at times t,
    from skyfield, and its rate of change. Both 3 x 3 x N (or 3 x 3 for one time).

    The rate keeps only the daily spin, d/dt R = -[w x] R with w along the Earth's axis;
    precession and nutation change R about a million times more slowly."""
    R = itrs.rotation_at(t)
    spin = OMEGA_EARTH * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return R, np.einsum("ij,jk...->ik...", -spin, R)


def sun_position_m(t):
    """Sun's position from the Earth's center (m), 3 x N, from the Astronomical Almanac's
    low-precision formula: good to about 0.01 deg, enough for Vizard's lighting and for
    the Sun's tiny pull on a low orbit (about 5e-7 m/s^2)."""
    n = np.atleast_1d(t.tt - 2451545.0)
    g = np.radians(357.528 + 0.9856003 * n)                  # mean anomaly
    lam = np.radians(280.460 + 0.9856474 * n + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)                      # obliquity of the ecliptic
    dist = (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)) * AU_M
    return dist * np.array([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])


class Environment(sysModel.SysModel):
    """A Basilisk module, written in Python, that publishes the Earth's orientation and
    the Sun's position every step, in the messages Basilisk's SPICE module would.

    Basilisk would normally read these from NASA's SPICE files. Taking them from skyfield
    instead puts the station exactly where Weeks 2 and 3 put it, so any difference from
    SGP4 comes from the orbit, not from the Earth's orientation."""

    def __init__(self, epoch):
        super().__init__()
        self.ModelTag = "environment"
        self.epoch = epoch                       # skyfield Time at simulation time 0
        self.earthOutMsg = messaging.SpicePlanetStateMsg()
        self.sunOutMsg = messaging.SpicePlanetStateMsg()
        self._chunk = None

    def Reset(self, current_nanos):
        self._chunk = None

    def _load_chunk(self, start_s):
        seconds = start_s + np.arange(0, CHUNK_S, STEP_S)
        t = dp.times_after(self.epoch, seconds)
        R, R_dot = earth_rotation(t)
        self._chunk = (start_s, R, R_dot, sun_position_m(t))

    def UpdateState(self, current_nanos):
        sim_s = current_nanos * 1e-9
        if self._chunk is None or not 0 <= sim_s - self._chunk[0] < CHUNK_S:
            self._load_chunk(np.floor(sim_s))
        start_s, R, R_dot, sun = self._chunk
        i = int(round((sim_s - start_s) / STEP_S))

        earth = messaging.SpicePlanetStateMsgPayload()
        earth.PlanetName = "earth"
        earth.J20002Pfix = R[:, :, i].tolist()
        earth.J20002Pfix_dot = R_dot[:, :, i].tolist()
        earth.computeOrient = 1
        self.earthOutMsg.write(earth, current_nanos, self.moduleID)

        sun_msg = messaging.SpicePlanetStateMsgPayload()
        sun_msg.PlanetName = "sun"
        sun_msg.PositionVector = sun[:, i].tolist()
        sun_msg.J20002Pfix = np.eye(3).tolist()
        self.sunOutMsg.write(sun_msg, current_nanos, self.moduleID)


class BeaconLight(sysModel.SysModel):
    """Marks the beacon as "sending" for Vizard while the station can hear the satellite.

    The real beacon is on all the time; lighting it up only during the pass shows when the
    station is receiving it."""

    def __init__(self, access_msg):
        super().__init__()
        self.ModelTag = "beaconLight"
        self.accessInMsg = messaging.AccessMsgReader()
        self.accessInMsg.subscribeTo(access_msg)
        self.stateOutMsg = messaging.DataNodeUsageMsg()

    def UpdateState(self, current_nanos):
        out = messaging.DataNodeUsageMsgPayload()
        out.baudRate = -BEACON_BAUD if self.accessInMsg().hasAccess else 0.0
        self.stateOutMsg.write(out, current_nanos, self.moduleID)


# --- The simulation -----------------------------------------------------------------------

def build_sim(sat, record, r0, v0, f0, viz_file=None, stream=False, realtime=False,
              degree=GRAV_DEGREE):
    """Set up the Basilisk simulation. Returns a dict of the simulation and its parts.

    Tasks: "dynamics" (environment, spacecraft, ground station, recorders), "viz"
    (Vizard, saving to viz_file or streaming live; only when one of those is asked for)
    and "clock" (keeps pace with the wall clock; only with realtime). The viz and clock
    tasks start disabled."""
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("process")
    step = macros.sec2nano(STEP_S)
    process.addTask(sim.CreateNewTask("dynamics", step))

    env = Environment(sat.epoch)
    sim.AddModelToTask("dynamics", env, 100)

    sc = spacecraft.Spacecraft()
    sc.ModelTag = sat.name.split(" (")[0].title().replace(" ", "-")      # "Umka-1"
    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True
    earth.useSphericalHarmonicsGravityModel(str(get_path(DataFile.LocalGravData.GGM03S)), degree)
    earth.planetBodyInMsg.subscribeTo(env.earthOutMsg)
    sun = gravity.createSun()
    sun.planetBodyInMsg.subscribeTo(env.sunOutMsg)
    gravity.addBodiesTo(sc)
    sc.hub.r_CN_NInit = r0.tolist()
    sc.hub.v_CN_NInit = v0.tolist()
    sim.AddModelToTask("dynamics", sc, 50)

    station_xyz = pp.make_station(record["lat"], record["lng"], record["altitude"]).itrs_xyz.m
    ground = groundLocation.GroundLocation()
    ground.ModelTag = f"SatNOGS-{record['id']}" if record["id"] else "station"
    ground.planetRadius = EARTH_RADIUS_M
    ground.specifyLocationPCPF(np.asarray(station_xyz).reshape(3, 1))   # WGS84, like skyfield
    ground.minimumElevation = np.radians(pp.MIN_ELEVATION_DEG)
    ground.planetInMsg.subscribeTo(env.earthOutMsg)
    ground.addSpacecraftToModel(sc.scStateOutMsg)
    sim.AddModelToTask("dynamics", ground, 10)

    beacon = BeaconLight(ground.accessOutMsgs[0])
    sim.AddModelToTask("dynamics", beacon, 5)

    state_rec = sc.scStateOutMsg.recorder()
    access_rec = ground.accessOutMsgs[0].recorder()
    sim.AddModelToTask("dynamics", state_rec)
    sim.AddModelToTask("dynamics", access_rec)

    parts = {"sim": sim, "env": env, "sc": sc, "ground": ground, "beacon": beacon,
             "state_rec": state_rec, "access_rec": access_rec, "viz": None,
             "station_xyz": np.asarray(station_xyz)}
    if viz_file or stream:
        parts["viz"] = add_vizard(sim, step, parts, sat, record, f0, viz_file, stream)
    if realtime:
        process.addTask(sim.CreateNewTask("clock", step))
        clock = simSynch.ClockSynch()
        clock.ModelTag = "clock"
        sim.AddModelToTask("clock", clock)
        parts["clock"] = clock
        sim.disableTask("clock")
    return parts


def add_vizard(sim, step, parts, sat, record, f0, viz_file, stream):
    """Add Vizard to the simulation: the station, its cone, the beacon and a camera.
    It runs in its own process so it always comes after the dynamics."""
    sim.CreateNewProcess("vizProcess", -1).addTask(sim.CreateNewTask("viz", step))

    beacon = vizInterface.Transceiver()
    beacon.r_SB_B = [0.0, 0.0, 0.0]
    beacon.normalVector = [0.0, 0.0, 1.0]
    beacon.fieldOfView = np.radians(170)
    beacon.label = f"{f0 / 1e6:.3f} MHz CW beacon"
    beacon.animationSpeed = 2
    beacon.transceiverStateInMsgs = messaging.DataNodeUsageMsgInMsgsVector(
        [parts["beacon"].stateOutMsg.addSubscriber()])

    viz = vizSupport.enableUnityVisualization(
        sim, "viz", parts["sc"], saveFile=str(viz_file) if viz_file else None,
        transceiverList=[[beacon]], liveStream=stream)
    parts["viz_beacon"] = beacon          # Vizard keeps only a pointer to it

    epoch = sat.epoch.utc_datetime()
    payload = messaging.EpochMsgPayload()
    payload.year, payload.month, payload.day = epoch.year, epoch.month, epoch.day
    payload.hours, payload.minutes = epoch.hour, epoch.minute
    payload.seconds = epoch.second + epoch.microsecond / 1e6
    parts["epoch_msg"] = messaging.EpochMsg().write(payload)      # keep a reference
    viz.epochInMsg.subscribeTo(parts["epoch_msg"])

    name = f"SatNOGS {record['id']}" if record["id"] else "Station"
    vizSupport.addLocation(viz, stationName=name, parentBodyName="earth",
                           r_GP_P=parts["station_xyz"].tolist(),
                           fieldOfView=np.radians(2 * (90 - pp.MIN_ELEVATION_DEG)),
                           range=3_000_000.0, color="orange")
    vizSupport.createStandardCamera(viz, setMode=0, setView=0, bodyTarget="earth",
                                    fieldOfView=np.radians(60),
                                    displayName=f"Looking down from {parts['sc'].ModelTag}")
    s = viz.settings
    s.showLocationCones = 1
    s.showLocationCommLines = 1
    s.showLocationLabels = 1
    s.showSpacecraftLabels = 1
    s.orbitLinesOn = 1
    s.trueTrajectoryLinesOn = 1
    s.spacecraftSizeMultiplier = 3
    s.show24hrClock = 1
    sim.disableTask("viz")
    return viz


def run_until(parts, stop_s):
    parts["sim"].ConfigureStopTime(macros.sec2nano(stop_s))
    parts["sim"].ExecuteSimulation()


# --- Reading the results ---------------------------------------------------------------------

def results(parts):
    """The recorded run as arrays: seconds since the TLE epoch, positions and velocities
    (3 x N, m and m/s), and the station's elevation (deg), range (m) and range-rate (m/s)
    from groundLocation."""
    s, a = parts["state_rec"], parts["access_rec"]
    return {"seconds": s.times() * 1e-9, "r": s.r_BN_N.T, "v": s.v_BN_N.T,
            "elevation_deg": np.degrees(a.elevation), "range_m": a.slantRange,
            "range_rate": a.range_dot, "has_access": a.hasAccess.astype(bool)}


def geodetic_elevation_deg(r_n, R, station_xyz, lat_deg, lon_deg):
    """Elevation (deg) of positions r_n (3 x N, inertial) above the station's horizon,
    measured from the WGS84 vertical like skyfield's altaz (Week 2).

    groundLocation measures from the line to the Earth's center instead; at 34 deg
    latitude the two verticals differ by about 0.19 deg."""
    r_itrs = np.einsum("ijn,jn->in", R, r_n)
    rho = r_itrs - station_xyz[:, None]
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return np.degrees(np.arcsin(up @ rho / np.linalg.norm(rho, axis=0)))


def passes_from_elevation(seconds, elevation, min_el=pp.MIN_ELEVATION_DEG):
    """Complete passes in a sampled elevation series, as dicts with rise, peak and set
    times (seconds, rise and set interpolated to where elevation crosses min_el) and the
    peak elevation. Like predict_passes.find_passes, a pass already up at the start or
    still up at the end is left out."""
    above = elevation > min_el
    edges = np.diff(above.astype(int))
    rises, sets = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]

    def cross(i):
        e0, e1 = elevation[i], elevation[i + 1]
        return seconds[i] + (min_el - e0) * (seconds[i + 1] - seconds[i]) / (e1 - e0)

    passes = []
    for i in rises:
        later = sets[sets > i]
        if not len(later):
            break
        j = later[0]
        k = i + 1 + int(np.argmax(elevation[i + 1:j + 1]))
        passes.append({"rise_s": cross(i), "set_s": cross(j), "peak_s": seconds[k],
                       "peak_deg": float(elevation[k])})
    return passes


def drag_lag_m(sat, seconds):
    """How far (m) drag moves the satellite ahead along its orbit in `seconds`, from the
    TLE's first derivative of mean motion.

    Drag lowers the orbit, so the satellite speeds up. The TLE stores half the rate of
    change of the mean motion (sgp4 keeps it as ndot, in rad/min^2), so after t minutes
    the satellite is 1/2 x (2 ndot) x t^2 = ndot x t^2 radians ahead, or ndot x t^2 x a
    meters for an orbit of radius a."""
    a = (om.MU_EARTH * 1e9 / (sat.model.no_kozai / 60) ** 2) ** (1 / 3)
    return sat.model.ndot * (seconds / 60) ** 2 * a


def nearest_pass(passes, rise_s):
    return min(passes, key=lambda q: abs(q["rise_s"] - rise_s)) if passes else None


# --- Plots -------------------------------------------------------------------------------------

def plot_drift(hours, drift_km, title, out_path):
    plt = dp._plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hours, drift_km, lw=1.5)
    ax.set_xlabel("Hours after the TLE epoch")
    ax.set_ylabel("Basilisk − SGP4 position (km)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_doppler_compare(seconds, shift_bsk, shift_sgp4, rise_label, title, out_path):
    plt = dp._plt()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    minutes = seconds / 60
    top.plot(minutes, shift_sgp4 / 1e3, lw=4, alpha=0.35, label="skyfield / SGP4 (Week 3)")
    top.plot(minutes, shift_bsk / 1e3, lw=1.5, ls="--", color="k", label="Basilisk")
    top.axhline(0, color="gray", lw=0.8)
    top.set_ylabel("Doppler shift (kHz)")
    top.legend(loc="upper right")
    top.grid(alpha=0.3)
    top.set_title(title)
    bottom.plot(minutes, shift_bsk - shift_sgp4)
    bottom.axhline(0, color="gray", lw=0.8)
    bottom.set_ylabel("Basilisk − SGP4 (Hz)")
    bottom.set_xlabel(f"Minutes after rise ({rise_label} UTC, from SGP4)")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --- Modes -------------------------------------------------------------------------------------

def simulate_pass(args, sat, record, station, t0):
    """Propagate from the TLE epoch through the highest pass in the window, compare with
    SGP4 and save the pass for Vizard."""
    f0 = args.freq_hz
    passes = pp.find_passes(sat, station, t0, t0 + args.days)
    if not passes:
        raise SystemExit("No passes above the minimum elevation in this window")
    best = max(passes, key=lambda q: q["peak_deg"])
    rise_s = (best["rise"] - sat.epoch) * 86400
    set_s = (best["set"] - sat.epoch) * 86400
    if rise_s - VIZ_MARGIN_S < 0:
        raise SystemExit("The pass starts before the TLE epoch; choose a later --start")
    print(f"Beacon f_0     : {f0 / 1e6:.3f} MHz")
    print(f"Chosen pass    : highest of {len(passes)} (SGP4), rise {pp.utc(best['rise'])} UTC, "
          f"peak {best['peak_deg']:.1f}°, {(set_s - rise_s) / 86400 * 1440:.1f} min, "
          f"{rise_s / 86400:.2f} days after the TLE epoch")

    tag = f"{sat.model.satnum}_{record['id'] or 'custom'}"
    viz_file = None if args.no_viz else VIZ_DIR / f"pass_{tag}.bin"
    if viz_file:
        viz_file.parent.mkdir(parents=True, exist_ok=True)
    r0, v0 = tle_state(tle_path(args), args.norad)
    parts = build_sim(sat, record, r0, v0, f0, viz_file=viz_file, degree=args.degree)
    sim = parts["sim"]
    sim.InitializeSimulation()

    tic = time.time()
    run_until(parts, rise_s - VIZ_MARGIN_S)
    if viz_file:
        sim.enableTask("viz")
    run_until(parts, set_s + VIZ_MARGIN_S)
    print(f"Basilisk run   : {(set_s + VIZ_MARGIN_S) / 86400:.2f} days in {time.time() - tic:.0f} s "
          f"({STEP_S} s steps, GGM03S {args.degree}x{args.degree} + Sun, no drag)")

    res = results(parts)
    seconds = res["seconds"]
    t = dp.times_after(sat.epoch, seconds)
    sgp4 = sat.at(t)
    drift = np.linalg.norm(res["r"] - sgp4.position.m, axis=0)
    start_diff = float(drift[0])
    print(f"Start          : Basilisk and SGP4 differ by {start_diff:.2f} m at the TLE epoch")
    print(f"Drift          : {np.interp(3600, seconds, drift) / 1e3:.2f} km after 1 h, "
          f"{np.interp(min(86400, seconds[-1]), seconds, drift) / 1e3:.2f} km after "
          f"{min(86400, seconds[-1]) / 3600:.0f} h, {drift[-1] / 1e3:.2f} km at the end")
    print(f"Drag estimate  : the TLE's mean-motion decay puts SGP4 "
          f"{drag_lag_m(sat, seconds[-1]) / 1e3:.2f} km further along its orbit by the end "
          f"(Basilisk has no drag)")

    # The pass, in Basilisk's orbit, with the Week 2 definition of elevation.
    R, R_dot = earth_rotation(t)
    elev = geodetic_elevation_deg(res["r"], R, parts["station_xyz"], record["lat"], record["lng"])
    bsk_pass = nearest_pass(passes_from_elevation(seconds, elev), rise_s)
    if bsk_pass is None:
        raise SystemExit("Basilisk's orbit has no pass near SGP4's; it has drifted too far")
    print(f"Basilisk pass  : rise {bsk_pass['rise_s'] - rise_s:+.1f} s, "
          f"set {bsk_pass['set_s'] - set_s:+.1f} s from SGP4's, "
          f"peak {bsk_pass['peak_deg']:.2f}° ({bsk_pass['peak_deg'] - best['peak_deg']:+.2f}°)")

    # Doppler: groundLocation's range-rate vs the Week 3 dot product, on the same times.
    on = (seconds >= rise_s) & (seconds <= set_s)
    s_on = seconds[on]
    station_r = parts["station_xyz"]
    r_sta = np.einsum("jin,j->in", R[:, :, on], station_r)                # ITRS -> inertial
    v_sta = np.einsum("jin,j->in", R_dot[:, :, on], station_r)
    _, rate_formula = dp.range_rate_dot(res["r"][:, on], res["v"][:, on], r_sta, v_sta)
    rate_bsk = res["range_rate"][on]
    formula_diff = float(np.max(np.abs(rate_bsk - rate_formula)))
    _, shift_bsk = dp.doppler_hz(f0, rate_bsk)
    week3 = dp.analyse_times(sat, station, s_on - rise_s, t[on], f0)
    shift_sgp4 = week3["shift"]
    diff_hz = shift_bsk - shift_sgp4
    print(f"Range-rate     : groundLocation vs Week 3 dot product on Basilisk's orbit: "
          f"largest difference {formula_diff * 1e3:.3f} mm/s")
    print(f"Doppler shift  : Basilisk {shift_bsk[0]:+.0f} Hz at rise -> {shift_bsk[-1]:+.0f} Hz at set "
          f"(SGP4 {shift_sgp4[0]:+.0f} -> {shift_sgp4[-1]:+.0f} Hz)")
    print(f"Basilisk − SGP4: largest |difference| {np.max(np.abs(diff_hz)):.1f} Hz "
          f"(typical {np.median(np.abs(diff_hz)):.1f} Hz)")

    problems = []
    if start_diff > START_MATCH_M:
        problems.append(f"starting states differ by {start_diff:.1f} m")
    if formula_diff > RANGE_RATE_MATCH_M_S:
        problems.append(f"groundLocation range-rate differs from the formula by {formula_diff:.3f} m/s")
    if not (shift_bsk[0] > 0 > shift_bsk[-1]):
        problems.append("Basilisk's shift should start positive and end negative")
    print("Checkpoint     : " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    where = f"SatNOGS station {record['id']}" if record["id"] else "the station"
    day = best["rise"].utc_strftime("%Y-%m-%d")
    hours = seconds[::60] / 3600
    print(f"Drift plot     : {plot_drift(hours, drift[::60] / 1e3, f'{sat.name}: Basilisk vs SGP4 from the same TLE', PLOT_DIR / f'basilisk_drift_{tag}.png')}")
    print(f"Doppler plot   : {plot_doppler_compare(s_on - rise_s, shift_bsk, shift_sgp4, pp.utc(best['rise']), f'{sat.name} over {where}, {day}: Basilisk vs SGP4', PLOT_DIR / f'basilisk_doppler_{tag}.png')}")
    if viz_file:
        print(f"Vizard file    : {viz_file}  (open it in Vizard: File > Load)")


def live(args, sat, record, t0):
    """Catch up from the TLE epoch to now, then run in step with the clock (times --speed)
    and stream to Vizard, printing where the satellite is every --print-every seconds."""
    f0 = args.freq_hz
    r0, v0 = tle_state(tle_path(args), args.norad)
    parts = build_sim(sat, record, r0, v0, f0, stream=not args.no_viz, realtime=True,
                      degree=args.degree)
    parts["clock"].accelFactor = args.speed
    sim = parts["sim"]
    if not args.no_viz:
        print("Vizard         : start Vizard, choose live streaming on its start screen and "
              "connect to tcp://localhost:5556; Basilisk waits until it connects")
    sim.InitializeSimulation()

    # Catch up with the wall clock (or with --start). Catching up takes a few seconds,
    # during which the wall clock moves on, so repeat until less than a step behind.
    sim_s = 0.0
    tic = time.time()
    while True:
        goal = t0 if args.start else fs.ts.now()
        behind = (goal - sat.epoch) * 86400 - sim_s
        if behind < STEP_S:
            break
        sim_s += np.floor(behind / STEP_S) * STEP_S
        run_until(parts, sim_s)
    print(f"Caught up      : {sim_s / 86400:.2f} days after the TLE epoch in {time.time() - tic:.0f} s")
    print(f"Running        : {args.speed:g} x real time for {args.minutes:g} minutes"
          + ("" if args.no_viz else ", streaming to Vizard (start Vizard in Live Streaming mode)"))
    for task in ("clock",) if args.no_viz else ("clock", "viz"):
        sim.enableTask(task)

    station = pp.make_station(record["lat"], record["lng"], record["altitude"])
    end_s = sim_s + args.minutes * 60 * args.speed
    try:
        while sim_s < end_s:
            sim_s += args.print_every * args.speed
            run_until(parts, sim_s)
            res = parts["state_rec"]
            r = res.r_BN_N[-1]
            t = dp.times_after(sat.epoch, sim_s)
            acc = parts["access_rec"]
            el = np.degrees(acc.elevation[-1])
            R, _ = earth_rotation(t)
            lat, lon, alt = ecef_to_geodetic(R @ r)
            line = (f"{pp.utc(t)} UTC  lat {lat:+6.2f}  lon {lon:+7.2f}  alt {alt / 1e3:5.1f} km  "
                    f"elev {el:+5.1f}°")
            if acc.hasAccess[-1]:
                line += f"  IN VIEW  Doppler {dp.doppler_hz(f0, acc.range_dot[-1])[1]:+7.0f} Hz"
            sgp4_el = pp.elevation_deg(sat, station, t)
            print(line + f"  (SGP4 elev {sgp4_el:+5.1f}°)", flush=True)
            parts["state_rec"].clear()
            parts["access_rec"].clear()
    except KeyboardInterrupt:
        print("Stopped")


def ecef_to_geodetic(xyz):
    """Latitude, longitude (deg) and height (m) on the WGS84 ellipsoid of an
    Earth-fixed position (m), by Bowring's method (good to millimeters in orbit)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    b = a * (1 - f)
    x, y, z = xyz
    p = np.hypot(x, y)
    th = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + e2 / (1 - e2) * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), p / np.cos(lat) - n


def main():
    p = argparse.ArgumentParser(description="Propagate the satellite with Basilisk and watch it in Vizard.")
    pp.add_pass_args(p)
    p.set_defaults(days=DAYS_AHEAD)
    p.add_argument("--freq-hz", type=float, default=fs.BEACON_HZ, help="beacon frequency f_0 (Hz)")
    p.add_argument("--degree", type=int, default=GRAV_DEGREE,
                   help="degree and order of the GGM03S gravity field (2 = J2 only)")
    p.add_argument("--no-viz", action="store_true", help="don't save or stream anything for Vizard")
    p.add_argument("--live", action="store_true", help="run in real time from now and stream to Vizard")
    p.add_argument("--speed", type=float, default=1.0, help="with --live: times faster than real time")
    p.add_argument("--minutes", type=float, default=30, help="with --live: how long to run (wall clock)")
    p.add_argument("--print-every", type=float, default=10,
                   help="with --live: seconds of wall clock between status lines")
    args = p.parse_args()

    sat, record, station, t0 = pp.load_inputs(args)
    if args.live:
        live(args, sat, record, t0)
    else:
        simulate_pass(args, sat, record, station, t0)


if __name__ == "__main__":
    main()
