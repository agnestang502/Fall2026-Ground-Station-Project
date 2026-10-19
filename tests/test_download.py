"""Offline checks for download_funcube.py (the SatNOGS API is faked)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_funcube as dl  # noqa: E402


def fake_obs(obs_id, minutes_ago, status="good", audio=True, norad=39444):
    start = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {
        "id": obs_id, "status": status, "norad_cat_id": norad,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (start + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload": f"https://example.org/{obs_id}.ogg" if audio else None,
        "waterfall": f"https://example.org/{obs_id}.png",
        "station_name": "Test GS", "ground_station": 1,
        "station_lat": 21.3, "station_lng": -157.8, "station_alt": 10,
        "tle0": "FUNCUBE-1", "tle1": "1 39444U ...", "tle2": "2 39444 ...",
    }


def test_selection_and_metadata(tmp_path, monkeypatch):
    page = [
        fake_obs(10, 10),
        fake_obs(11, 20),                                # too close to #10 -> skipped
        fake_obs(12, 60, audio=False),                   # no audio -> skipped
        fake_obs(18, 80, norad=43017),                   # another satellite -> skipped
        fake_obs(13, 100),
        fake_obs(14, 200, status="bad"),                 # wrong status -> skipped
        fake_obs(15, 300),
        fake_obs(16, 60 * 30),                           # older than 24 h -> stop
        fake_obs(17, 60 * 31),
    ]
    urls = []
    monkeypatch.setattr(dl, "get_page", lambda url: (urls.append(url), (page, None))[1])
    monkeypatch.setattr(dl, "download", lambda url, dest: dest.write_bytes(b"ogg"))
    monkeypatch.setattr(dl, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["download_funcube.py", "--skip-validation"])

    dl.main()

    out = tmp_path / str(dl.NORAD_ID)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [m["id"] for m in manifest] == [10, 13, 15]
    assert manifest[0]["station_lat"] == 21.3 and manifest[0]["tle1"].startswith("1 39444")
    assert json.loads((out / "obs_13.json").read_text())["tle2"].startswith("2 39444")
    assert (out / "obs_15.ogg").exists()
    assert (out / "obs_15_waterfall.png").exists()
    assert manifest[0]["waterfall_file"] == "obs_10_waterfall.png"
    assert "norad_cat_id=39444" in urls[0] and "satellite__" not in urls[0]
