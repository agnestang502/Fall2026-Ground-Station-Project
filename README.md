# Fall2026-Ground-Station-Project

Estimating satellite position and velocity from SatNOGS ground-station recordings,
by comparing the observed signal frequency with what each station expected from its TLE.

## Pipeline so far

1. **`download_funcube.py`**: downloads recent passes for a satellite (default
   FUNcube-1, NORAD 39444, last 24 h) with the audio, the SatNOGS waterfall image **and** metadata
   (ground-station location and the TLE the station used).
2. **`validate_signals.py`**: runs automatically after downloading. It first screens the
   waterfall image with the CNN classifier in **`waterfall_cnn.py`** (by Tata Li), then
   checks the audio for FUNcube-1's signal, and rejects anything doubtful.

See [docs/signal_validation.md](docs/signal_validation.md) for a plain-English
explanation and the results on our first 10 passes.

## Simulating a satellite pass

A five-week plan to simulate a pass over a ground station (orbit, Doppler curve,
waterfall), ending with a training-data generator.

- **Week 1** ([results](docs/week1_find_satellite.md)): the satellite is
  **UmKA-1** (NORAD 57172), with a CW beacon at **437.625 MHz**.
  - `pick_satellite.py` ranks satellites with an active CW beacon on SatNOGS DB by how
    often SatNOGS Network has heard them recently.
  - `find_satellite.py` fetches the TLE from CelesTrak (saved to `data/tle/`, reused if
    offline), prints its epoch and age and where the satellite is right now, and checks
    the Week 1 checkpoint. `--ground-track` plots one orbit.

```bash
python pick_satellite.py                 # rank CW satellites (a few minutes)
python find_satellite.py --ground-track  # UmKA-1 by default
python find_satellite.py --norad <NORAD> --freq-hz <beacon Hz>
```

## Running

```bash
pip install -r requirements.txt
python download_funcube.py            # download + validate
python -m pytest tests/               # run the tests
```
