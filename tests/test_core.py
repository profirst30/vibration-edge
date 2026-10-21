import json
import shutil
import subprocess

import numpy as np
import pytest

from src.features import FS, features
from src.payload import decode, encode
from src.radio import message_airtime_s, phy_airtime_s


def test_airtime_matches_datasheet_example():
    assert phy_airtime_s(10, 7) == pytest.approx(0.041216)  # 10 bytes, SF7, 125 kHz: 41.216 ms
    assert phy_airtime_s(64, 12) == pytest.approx(2.793472)  # 51-byte payload at SF12, the slowest uplink


def test_long_message_is_split():
    assert message_airtime_s(300, 7) > 0
    assert message_airtime_s(52, 12) > 2 * phy_airtime_s(13, 12)


def test_features_of_a_sine():
    t = np.arange(FS) / FS
    f = features(np.sqrt(2) * np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert f[0] == pytest.approx(1, rel=1e-3)  # RMS of an amplitude-sqrt(2) sine
    assert f[2] == pytest.approx(np.sqrt(2), rel=1e-3)  # crest factor
    assert f[3] == pytest.approx(1.5, rel=1e-3)  # kurtosis of a sine
    assert np.sum(f[4:] ** 2) == pytest.approx(f[0] ** 2, rel=1e-3)  # bands add up to the RMS (Parseval)
    assert np.argmax(f[4:]) == 0  # 1 kHz falls in the first 1.25 kHz band


def test_frame_round_trip():
    feats = np.array([0.07, 0.46, 6.3, 3.6, 0.04, 0.02, 0.03, 0.04, 0.02, 0.01, 0.01, 0.008], dtype=np.float32)
    frame = encode(1, feats)
    assert len(frame) == 26
    out = decode(frame)
    assert out["status"] == "alert"
    assert np.allclose(out["features"], feats, rtol=3e-3)
    assert decode(encode(0)) == {"status": "normal"}
    with pytest.raises(ValueError):
        decode(b"\x02\x00")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_decoder_matches_python():
    feats = np.array([0.07, 0.46, 6.3, 3.6, 0.04, 0.02, 0.03, 0.04, 0.02, 0.01, 0.01, 0.008], dtype=np.float32)
    frame = list(encode(0, feats))
    js = f"const d=require('./docs/decoder.js');console.log(JSON.stringify(d.decodeUplink({{bytes:{frame}}})))"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out["data"]["status"] == "normal"
    assert np.allclose(list(out["data"]["features"].values()), decode(bytes(frame))["features"])
