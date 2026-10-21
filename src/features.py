"""On-node features of one vibration snapshot, extracted once for the three IMS run-to-failure tests.

    python -m src.features        # reads data/<test>/, writes data/features.npz (a few minutes)

Each snapshot is 1 s at 20 kHz (20,480 samples). The 12 features follow the classic signs of wear
for machine health: overall level (RMS, peak), new frequency components (8 band levels) and
intermittency (crest factor, kurtosis).
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "features.npz"
FS = 20_000
N_BANDS = 8
NAMES = ["rms", "peak", "crest", "kurtosis"] + [f"band{b}" for b in range(N_BANDS)]
# Folder, columns kept (one accelerometer per bearing, like a single wireless sensor), and the bearings that
# failed at the end of the test, from the IMS readme. Test 1 has x and y axes: we keep x (ch 1, 3, 5, 7).
TESTS = {
    "test1": ("1st_test", [0, 2, 4, 6], [3, 4]),
    "test2": ("2nd_test", [0, 1, 2, 3], [1]),
    "test3": ("4th_test/txt", [0, 1, 2, 3], [3]),
}


def features(x: np.ndarray) -> np.ndarray:
    """12 features of one snapshot (float32, any length)."""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    kurt = np.mean(x ** 4) / rms ** 4
    power = np.abs(np.fft.rfft(x)) ** 2
    bands = np.array_split(power[1:], N_BANDS)  # equal-width bands from 0 to FS/2, DC excluded
    band_rms = np.sqrt([2 * b.sum() for b in bands]) / len(x)  # Parseval: the band's share of the RMS
    return np.array([rms, peak, peak / rms, kurt, *band_rms], dtype=np.float32)


def timestamp(name: str) -> datetime:
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def extract() -> None:
    out = {}
    with ProcessPoolExecutor() as pool:
        for test, (folder, cols, failed) in TESTS.items():
            files = sorted((DATA / folder).iterdir())
            n_cols = 8 if test == "test1" else 4
            x = np.stack(list(pool.map(_read, [(f, cols, n_cols) for f in files], chunksize=16)))
            t0 = timestamp(files[0].name)
            out[f"{test}_hours"] = np.array([(timestamp(f.name) - t0).total_seconds() / 3600 for f in files])
            out[f"{test}_features"] = x  # (snapshots, bearings, 12)
            out[f"{test}_failed"] = np.array(failed)
            print(f"{test}: {len(files)} snapshots, {len(cols)} bearings")
    np.savez_compressed(CACHE, **out)


def _read(args: tuple[Path, list[int], int]) -> np.ndarray:
    path, cols, n_cols = args
    x = np.fromstring(path.read_text(), dtype=np.float32, sep=" ").reshape(-1, n_cols)
    return np.stack([features(x[:, c]) for c in cols])


def load() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        extract()
    with np.load(CACHE) as f:
        return dict(f)


if __name__ == "__main__":
    extract()
