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
3. **`track_peaks.py`**: for every pass that passed the check, follows the satellite's
   frequency second by second and measures the **residual** (what we heard minus
   what the station expected), with a plot per pass.

Plain-English explanations and results:
[docs/signal_validation.md](docs/signal_validation.md) (signal check) and
[docs/peak_tracking.md](docs/peak_tracking.md) (peak tracking and residuals).

```bash
pip install -r requirements.txt
python download_funcube.py            # download + validate
python track_peaks.py data/39444      # track the signal, plot residuals
python -m pytest tests/               # run the tests
```
