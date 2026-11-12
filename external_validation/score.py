"""Scor prediksjons-CSV-er mot fasiten med arrangørens evaluation/possession_metrics.py.

Prediksjonene ligger i én mappe, én fil per kamp: <mappe>/<kamp>.csv med header frame,label.

Kjør fra repo-mappen:
  python external_validation/score.py ext_data/preds/baseline_w50
  python external_validation/score.py ext_data/preds/baseline_w50 --matches J03WPY J03WMX
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def load_metrics():
    """Importer arrangørens possession_metrics.py (ingen kopi - alltid deres versjon)."""
    path = C.EVALUATION_DIR / "possession_metrics.py"
    if not path.exists():
        sys.exit("Finner ikke {}. Kjør skriptet fra lagets repo (kopien av arrangørens repo).".format(path))
    sys.path.insert(0, str(C.EVALUATION_DIR))
    import possession_metrics as pm
    return pm


PM = load_metrics()


def built_matches():
    """Alle kamper som er bygget (har ext_data/<kamp>/gt.csv)."""
    if not C.OUT_DIR.exists():
        return []
    return sorted(p.name for p in C.OUT_DIR.iterdir() if (p / "gt.csv").exists())


def load_gt(match_id):
    path = C.OUT_DIR / match_id / "gt.csv"
    if not path.exists():
        sys.exit("Mangler fasit for {}. Kjør build_dataset.py først.".format(match_id))
    return PM.read_label_csv(path)


def score_labels(gt_frame, pred):
    """Scor én kamp. pred kan være en liste/array med etiketter eller en DataFrame frame,label."""
    if isinstance(pred, pd.DataFrame):
        pred_frame = pred
    else:
        labels = np.asarray(pred, dtype=object)
        pred_frame = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    problem = PM.validate_submission(pred_frame, len(gt_frame))
    if problem:
        raise ValueError("Ugyldig prediksjon: " + problem)
    res = PM.evaluate(PM.label_codes(gt_frame), PM.label_codes(pred_frame))
    return {
        "spell_f1": res["spell_f1_mean"],
        "macro_f1": res["macro_f1"],
        "n_gt_spells": res["n_gt_spells"],
        "n_pred_spells": res["n_pred_spells"],
    }


def score_dir(pred_dir, matches):
    pred_dir = Path(pred_dir)
    rows = []
    for m in matches:
        path = pred_dir / "{}.csv".format(m)
        if not path.exists():
            print("  (hopper over {}: finner ikke {})".format(m, path))
            continue
        row = {"match": m}
        row.update(score_labels(load_gt(m), PM.read_label_csv(path)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir", help="Mappe med <kamp>.csv-filer")
    ap.add_argument("--matches", nargs="+", default=None, help="Standard: alle bygde kamper")
    args = ap.parse_args()

    matches = args.matches or built_matches()
    table = score_dir(args.pred_dir, matches)
    if table.empty:
        sys.exit("Ingen prediksjoner å score.")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))
    print("\nSnitt spellF1 = {:.4f}   snitt macroF1 = {:.4f}   ({} kamper)".format(
        table["spell_f1"].mean(), table["macro_f1"].mean(), len(table)))


if __name__ == "__main__":
    main()
