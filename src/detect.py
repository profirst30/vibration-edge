"""Two anomaly detectors that could run on the node, both learning only from the sensor's first days.

- z-score: each log-feature is compared to its own baseline mean and spread; score = largest |z|.
- autoencoder: a tiny network rebuilds the 12 standardised features; score = reconstruction error.
  Exported to ONNX and quantized to INT8, the format an embedded runtime would load.

Same alert rule for both: threshold = MARGIN x the highest score seen during the baseline, and an
alert only after PERSIST snapshots in a row above it.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .features import ROOT

BASELINE = 0.10  # first 10 % of each test: the node learns its own normal after installation
PERSIST = 3      # 3 snapshots = 30 min above the threshold before raising an alert
MARGIN = 1.5
MODELS = ROOT / "models"
FP32, INT8 = MODELS / "autoencoder_fp32.onnx", MODELS / "autoencoder_int8.onnx"


def standardize(f: np.ndarray, n_base: int) -> np.ndarray:
    """(snapshots, 12) raw features -> log-features standardised on the first n_base snapshots."""
    x = np.log10(f)
    mu, sd = x[:n_base].mean(0), x[:n_base].std(0) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def zscore(z: np.ndarray) -> np.ndarray:
    return np.abs(z).max(1)


def first_alert(score: np.ndarray, n_base: int) -> tuple[float, int | None]:
    """Threshold and index of the first snapshot that completes PERSIST in a row above it."""
    thr = MARGIN * score[:n_base].max()
    run = 0
    for i in range(n_base, len(score)):
        run = run + 1 if score[i] > thr else 0
        if run == PERSIST:
            return float(thr), i
    return float(thr), None


def alert_onsets(score: np.ndarray, thr: float, n_base: int) -> int:
    """Number of times the alert switches on (each one would send a frame in the alert-only strategy)."""
    above = score[n_base:] > thr
    run, on, onsets = 0, False, 0
    for a in above:
        run = run + 1 if a else 0
        if run == PERSIST and not on:
            on, onsets = True, onsets + 1
        if not a:
            on = False
    return onsets


def train_autoencoder(x: np.ndarray, epochs: int = 3000) -> torch.nn.Module:
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(12, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3), torch.nn.ReLU(),
                              torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 12))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t = torch.from_numpy(x)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(t) - t) ** 2)
        loss.backward()
        opt.step()
    return net.eval()


class _Calib(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.rows = iter(x[i:i + 64] for i in range(0, len(x), 64))

    def get_next(self):
        b = next(self.rows, None)
        return None if b is None else {"features": b}


def export(net: torch.nn.Module, calib: np.ndarray) -> None:
    MODELS.mkdir(exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 12), str(FP32), input_names=["features"], output_names=["rebuilt"],
                      dynamic_axes={"features": {0: "n"}, "rebuilt": {0: "n"}}, opset_version=17, dynamo=False)
    pre = MODELS / "autoencoder_pre.onnx"
    quant_pre_process(str(FP32), str(pre))
    # reduce_range: 7-bit weights avoid the U8S8 int16 saturation of x86 CPUs without VNNI (see onboard-inference-gate)
    quantize_static(str(pre), str(INT8), _Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, reduce_range=True)
    pre.unlink()


def session(path: Path) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def ae_score(sess: ort.InferenceSession, z: np.ndarray) -> np.ndarray:
    rebuilt = sess.run(None, {"features": z})[0]
    return np.mean((rebuilt - z) ** 2, axis=1)
