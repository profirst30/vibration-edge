"""Binary uplink frames sent by the node, and their decoding (the JavaScript twin is docs/decoder.js).

Byte 0: format version. Byte 1: status (0 = normal, 1 = alert). Then, in a feature frame only, the 12
features as big-endian int16, each the log10 of the feature times 1000 (0.1 % resolution).
"""
import struct

import numpy as np

VERSION = 1
SCALE = 1000
N_FEATURES = 12


def encode(status: int, feats: np.ndarray | None = None) -> bytes:
    head = bytes([VERSION, status])
    if feats is None:
        return head
    q = np.clip(np.round(np.log10(feats) * SCALE), -32768, 32767).astype(">i2")
    return head + q.tobytes()


def decode(frame: bytes) -> dict:
    if len(frame) < 2 or frame[0] != VERSION:
        raise ValueError("unknown frame format")
    out = {"status": "alert" if frame[1] else "normal"}
    if len(frame) == 2:
        return out
    if len(frame) != 2 + 2 * N_FEATURES:
        raise ValueError(f"feature frame of {len(frame)} bytes, expected {2 + 2 * N_FEATURES}")
    q = struct.unpack(f">{N_FEATURES}h", frame[2:])
    out["features"] = [10 ** (v / SCALE) for v in q]
    return out
