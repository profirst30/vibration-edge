"""Whole pipeline: detection on the three tests, INT8 model, radio budget, cost on this CPU.

    python -m src.run     # writes results/results.json, docs/results.json, docs/series.json, README table

Every number in the README and on the demo page comes from this script.
"""
import json
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

from . import detect, radio
from .features import FS, NAMES, ROOT, TESTS, features, load
from .payload import encode

RESULTS = ROOT / "results" / "results.json"
SNAPSHOTS_PER_DAY = 144  # one 1-second snapshot every 10 minutes, as in the IMS tests
RAW_BYTES = 20_480 * 2   # one snapshot as 16-bit samples
STOPPED = 0.2            # RMS under 20 % of the baseline median: machine stopped, not an anomaly
SERIES_POINTS = 400


def downsample(h: np.ndarray, *ys: np.ndarray) -> list[list[float]]:
    """At most SERIES_POINTS points per series, keeping the max of each window so peaks stay visible."""
    idx = np.array_split(np.arange(len(h)), min(SERIES_POINTS, len(h)))
    return [[round(float(h[i[-1]]), 2) for i in idx]] + [[round(float(y[i].max()), 4) for i in idx] for y in ys]


def cpu_name() -> str:
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True)
        return r.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def bench_us(fn, n: int = 200) -> float:
    for _ in range(10):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return float(np.median(t) * 1e6)


def main() -> None:
    d = load()
    sensors, base_rows = [], []
    for test, (_, _, failed) in TESTS.items():
        hours, feats = d[f"{test}_hours"], d[f"{test}_features"]
        n_base = int(detect.BASELINE * len(hours))
        for b in range(feats.shape[1]):
            f = feats[:, b]
            running = f[:, 0] > STOPPED * np.median(f[:n_base, 0])
            z = detect.standardize(f[running], n_base)
            sensors.append({"test": test, "bearing": b + 1, "failed": b + 1 in failed, "hours": hours[running],
                            "z": z, "n_base": n_base, "stopped": int((~running).sum())})
            base_rows.append(z[:n_base])

    base = np.concatenate(base_rows)
    net = detect.train_autoencoder(base)
    detect.export(net, base)
    sess = {"fp32": detect.session(detect.FP32), "int8": detect.session(detect.INT8)}

    rows, series = [], {}
    for s in sensors:
        scores = {"zscore": detect.zscore(s["z"]), **{f"ae_{k}": detect.ae_score(v, s["z"]) for k, v in sess.items()}}
        end = float(s["hours"][-1])
        row = {k: s[k] for k in ("test", "bearing", "failed", "stopped")} | {"snapshots": len(s["hours"]), "duration_h": round(end, 1)}
        for name, sc in scores.items():
            thr, i = detect.first_alert(sc, s["n_base"])
            row[name] = {"threshold": round(thr, 4), "first_alert_h": None if i is None else round(float(s["hours"][i]), 1),
                         "lead_h": None if i is None else round(end - float(s["hours"][i]), 1),
                         "onsets": detect.alert_onsets(sc, thr, s["n_base"])}
        rows.append(row)
        h, zs, ae = downsample(s["hours"], scores["zscore"], scores["ae_int8"])
        series[f"{s['test']}_b{s['bearing']}"] = {"hours": h, "zscore": zs, "ae_int8": ae, "failed": s["failed"],
                                                  "thr_zscore": row["zscore"]["threshold"], "thr_ae_int8": row["ae_int8"]["threshold"],
                                                  "alert_zscore": row["zscore"]["first_alert_h"], "alert_ae_int8": row["ae_int8"]["first_alert_h"]}

    # Radio: one sensor, one day, each strategy. Alert-only sends a daily heartbeat plus one frame per alert onset.
    days = sum(r["duration_h"] for r in rows) / 24
    onsets_per_day = sum(r["zscore"]["onsets"] for r in rows) / days
    feature_frame, status_frame = len(encode(0, np.ones(12, np.float32))), len(encode(0))
    strategies = {"raw": (SNAPSHOTS_PER_DAY, RAW_BYTES), "features": (SNAPSHOTS_PER_DAY, feature_frame),
                  "alert_only": (1 + onsets_per_day, status_frame)}
    radio_rows = []
    for sf in (7, 12):
        for name, (msgs, size) in strategies.items():
            air = msgs * radio.message_airtime_s(size, sf)
            radio_rows.append({"strategy": name, "sf": sf, "bytes_per_message": size, "messages_per_day": round(msgs, 2),
                               "uplinks_per_day": round(msgs * radio.frames(size, sf), 1), "airtime_s_per_day": round(air, 3),
                               "duty_cycle_share": round(air / (86_400 * radio.DUTY_CYCLE), 4)})

    # Cost of one decision on this CPU, single thread.
    x = np.random.default_rng(0).standard_normal(20_480).astype(np.float32) * 0.08
    z1 = sensors[0]["z"][:1]
    macs = sum(m.in_features * m.out_features for m in net if hasattr(m, "in_features"))
    cost = {"features_us": round(bench_us(lambda: features(x)), 1),
            **{f"ae_{k}_us": round(bench_us(lambda v=v: v.run(None, {"features": z1})), 1) for k, v in sess.items()},
            "ae_parameters": sum(p.numel() for p in net.parameters()), "ae_macs": macs,
            "ae_fp32_bytes": detect.FP32.stat().st_size, "ae_int8_bytes": detect.INT8.stat().st_size}

    out = {"machine": {"cpu": cpu_name(),"os": platform.platform(), "onnxruntime": ort.__version__},
           "settings": {"baseline": detect.BASELINE, "persist": detect.PERSIST, "margin": detect.MARGIN, "fs": FS,
                        "features": NAMES, "snapshots_per_day": SNAPSHOTS_PER_DAY},
           "sensors": rows, "radio": radio_rows, "cost": cost,
           "frames": {"feature_bytes": feature_frame, "status_bytes": status_frame, "raw_bytes": RAW_BYTES,
                      "sample_hex": encode(0, d["test2_features"][0, 0]).hex()}}  # first snapshot of test 2, bearing 1
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    shutil.copyfile(RESULTS, ROOT / "docs" / "results.json")
    (ROOT / "docs" / "series.json").write_text(json.dumps(series, separators=(",", ":")))
    readme = ROOT / "README.md"
    if readme.exists():
        text = re.sub(r"<!-- results -->.*<!-- /results -->", lambda _: f"<!-- results -->\n{table(out)}\n<!-- /results -->",
                      readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8")
    print(table(out))


STRATEGY_FR = {"raw": "signal brut", "features": "indicateurs", "alert_only": "alerte seule"}


def fr(x: float) -> str:
    return f"{x:,}".replace(",", " ").replace(".", ",")


def table(out: dict) -> str:
    def cell(r, k):
        a = r[k]
        return "aucune alerte" if a["first_alert_h"] is None else f"{fr(a['lead_h'])} h avant la fin ({a['onsets']} déclenchement{'s' * (a['onsets'] > 1)})"
    lines = ["| essai | roulement | cassé à la fin | z-score | autoencodeur FP32 | autoencodeur INT8 |", "|---|---:|---|---|---|---|"]
    lines += [f"| {r['test']} | {r['bearing']} | {'**oui**' if r['failed'] else 'non'} | {cell(r, 'zscore')} | {cell(r, 'ae_fp32')} | {cell(r, 'ae_int8')} |"
              for r in out["sensors"]]
    lines += ["", "| stratégie | SF | octets par message | trames par jour | temps d'antenne par jour (s) | part du rapport cyclique de 1 % |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {STRATEGY_FR[r['strategy']]} | {r['sf']} | {fr(r['bytes_per_message'])} | {fr(r['uplinks_per_day'])} | {fr(r['airtime_s_per_day'])} | {fr(round(100 * r['duty_cycle_share'], 2))} % |"
              for r in out["radio"]]
    c = out["cost"]
    lines += ["", f"Sur {out['machine']['cpu']}, un thread : indicateurs d'un enregistrement {fr(c['features_us'])} µs ; "
              f"autoencodeur FP32 {fr(c['ae_fp32_us'])} µs, INT8 {fr(c['ae_int8_us'])} µs ({c['ae_parameters']} paramètres, "
              f"{c['ae_macs']} multiplications-additions ; fichier ONNX de {fr(c['ae_fp32_bytes'])} octets en FP32, {fr(c['ae_int8_bytes'])} en INT8)."]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
