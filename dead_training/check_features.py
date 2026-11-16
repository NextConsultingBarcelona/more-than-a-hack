"""Check that the run-time features (src/dead_model.py, numpy only) equal the training-time
features (dead_training/deadball/features.py, which uses scipy) on a synthetic match.

    python dead_training/check_features.py

The two differ only in rounding: the run-time code avoids every operation that is not rounded
identically on all machines.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "tools")]
import make_fake_match                                    # noqa: E402
from deadball import features as train_features           # noqa: E402
from deadball.frames import Frames                         # noqa: E402
from src import dead_model, io as mio                      # noqa: E402

PITCH = (105.3, 68.0)


def main():
    out = tempfile.mkdtemp()
    sim = make_fake_match.simulate(seed=1, half_s=120.0)
    paths = make_fake_match.write(sim, out)
    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"])
    X, names = dead_model.features(trk, PITCH)
    L, W = PITCH
    fr = Frames(px=((trk.x - 0.5) * L).astype(np.float32), py=((trk.y - 0.5) * W).astype(np.float32),
                team=trk.team.astype(np.int8), bx=((trk.ball[:, 0] - 0.5) * L).astype(np.float32),
                by=((trk.ball[:, 1] - 0.5) * W).astype(np.float32), pitch=PITCH, fps=25)
    Y, names2 = train_features.build(fr)
    same_nan = bool(np.array_equal(np.isnan(X), np.isnan(Y)))
    diff = float(np.nanmax(np.abs(X - Y)))
    print("features: {}  same names: {}  same missing pattern: {}  largest difference: {:.2e}".format(
        len(names), names == names2, same_nan, diff))
    ok = names == names2 and same_nan and diff < 1e-3
    print("OK" if ok else "FEIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
