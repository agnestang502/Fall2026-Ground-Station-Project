# Fall2026-Ground-Station-Project

Estimating satellite position and velocity from SatNOGS ground-station recordings,
by comparing the observed signal frequency with what each station expected from its TLE.

## Pipeline so far

1. **`download_funcube.py`**: downloads recent passes for a satellite (default
   FUNcube-1, NORAD 39444, last 24 h) with the audio **and** metadata (ground-station
   location and the TLE the station used).
2. **`validate_signals.py`**: runs automatically after downloading. It checks each
   recording for a real satellite signal and rejects anything doubtful.

See [docs/signal_validation.md](docs/signal_validation.md) for a plain-English
explanation and the results on our first 10 passes.

```bash
pip install -r requirements.txt
python download_funcube.py            # download + validate
python -m pytest tests/               # run the tests
```
