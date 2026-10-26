# Week 1: pick a satellite and find it

## The satellite: UmKA-1

| | |
|---|---|
| Satellite | UmKA-1 (RS40S) |
| NORAD catalog number | **57172** |
| Beacon | "Mode U - CW", **f_0 = 437.625 MHz** (SatNOGS DB transmitter, active) |
| Orbit | sun-synchronous, inclination 97.49°, 491–509 km, period 94.6 min |
| Recent observations | 25+ "good" observations on SatNOGS Network in the 7 days to 2026-10-25 |

**How it was chosen.** `pick_satellite.py` took every transmitter on SatNOGS DB with
mode CW, marked active and alive, on a satellite listed as in orbit (185 of them). It kept
the ones on the 2 m or 70 cm amateur-satellite sub-bands, in 300–1,000 km orbits (52 of
them). Then it counted each one's good observations on SatNOGS Network over the last 7 days
(20 satellites checked; the API is rate-limited). Ten satellites reached the 25-observation
cap. UmKA-1 came first because its beacon is the highest frequency among them, which gives
the biggest Doppler shift (about ±10 kHz). That suits the plan's ±12 kHz waterfall grid.

Other good choices from the same run, all with 25+ good observations:

| NORAD | Satellite | f_0 (MHz) | Orbit (km) |
|---|---|---|---|
| 32791 | SEEDS II | 437.485 | 453–457 |
| 33499 | KKS-1 | 437.385 | 614–623 |
| 40908 | LilacSat-2 | 437.200 | 395–405 |
| 61746 | Colibri-S | 436.835 | 402–413 |
| 60237 | GRBBeta | 436.785 | 530–559 |
| 59112 | SONATE-2 | 145.840 | 354–360 |

SONATE-2 is a good **second satellite on another band** (2 m) for Week 5's tests.

Caveat: a "good" SatNOGS observation means *some* signal from that satellite was heard,
not necessarily the CW beacon. Before Week 4, open a few UmKA-1 waterfalls on SatNOGS
Network and check that the CW line is visible.

## Where it is

`python find_satellite.py`, run on 2026-10-25 with a CelesTrak TLE:

```
Satellite      : UMKA 1 (RS40S) (NORAD 57172)
Beacon f_0     : 437.625 MHz
TLE line 1     : 1 57172U 23091G   26298.71045447  .00003759  00000+0  16711-3 0  9998
TLE line 2     : 2 57172  97.4896 356.9021 0012962  30.6783 329.5208 15.22062440183850
TLE epoch      : 2026-10-25 17:03:03 UTC
TLE age        : 0.27 days
Inclination    : 97.49 deg
Period         : 94.6 min
Now            : 2026-10-25 23:33:15 UTC
Latitude       : +43.6902 deg
Longitude      : -37.6827 deg
Altitude       : 497.6 km
Checkpoint     : PASS
```

**Checkpoint.** The altitude, 497.6 km, is within 300–1,000 km. The orbit is tilted past
90°, so the latitude limit is 180° − 97.49° = 82.51°, and +43.69° is inside it.

## Stretch: ground track for one orbit

![UmKA-1 ground track](img/ground_track_umka1.png)

The track touches the ±82.5° limit (dashed) and never crosses it. The orbit ends about 24°
west of where it started, because Earth turns 360° × 94.6 / 1436 ≈ 23.7° underneath it in
one orbit.

## Running it

```bash
python pick_satellite.py                 # rank CW satellites (takes a few minutes)
python find_satellite.py --ground-track  # UmKA-1 by default
```

If CelesTrak or SatNOGS can't be reached from your machine, the
`Week 1 - pick and find the satellite` GitHub Actions workflow runs both scripts. It runs
whenever either script changes, and once it's on `main` you can also start it from the
Actions tab, optionally with another NORAD ID. The results are in the job log and the
`week1-results` artifact.
