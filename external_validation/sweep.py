"""Test mange parameterverdier og få spellF1 per kamp i én tabell.

Alt du skal endre står i en liten JSON-fil (se grids/example_baseline.json):
  {
    "model": "external_validation.example_baseline:predict",
    "grid":  {"window": [1, 25, 50, 125]}
  }
- "model": modul:funksjon. Funksjonen må ha formen predict(tracking, slots, **params)
  og returnere én etikett (H/A/D/X) per rad i tracking.
- "grid": hver parameter med en liste verdier. Alle kombinasjoner testes.

Kjør fra repo-mappen:
  python external_validation/sweep.py external_validation/grids/example_baseline.json
  python external_validation/sweep.py min_grid.json --matches J03WPY J03WMX

Velg verdien som er god på de FLESTE kampene (høyt snitt, lav spredning, OK minimum),
ikke den som vinner på én kamp. Tallene sammenligner parametere med hverandre -
de er ikke et anslag på konkurransescoren (Sportec definerer possession litt annerledes).
"""
import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from score import built_matches, load_gt, score_labels  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT))  # så "external_validation.x" og lagets egne moduler kan importeres


def load_model(spec):
    module_name, _, func_name = spec.partition(":")
    if not func_name:
        sys.exit("'model' må ha formen modul:funksjon, f.eks. external_validation.example_baseline:predict")
    return getattr(importlib.import_module(module_name), func_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_file", help="JSON-fil med 'model' og 'grid'")
    ap.add_argument("--matches", nargs="+", default=None, help="Standard: alle bygde kamper")
    args = ap.parse_args()

    spec = json.loads(Path(args.grid_file).read_text(encoding="utf-8"))
    predict = load_model(spec["model"])
    grid = spec["grid"]
    names = list(grid)
    combos = [dict(zip(names, values)) for values in itertools.product(*(grid[n] for n in names))]

    matches = args.matches or built_matches()
    if not matches:
        sys.exit("Ingen bygde kamper. Kjør build_dataset.py først.")
    print("Modell {} | {} kombinasjoner x {} kamper".format(spec["model"], len(combos), len(matches)))

    rows = []
    for m in matches:
        tracking = pd.read_parquet(C.OUT_DIR / m / "tracking.parquet")
        slots = pd.read_csv(C.OUT_DIR / m / "slots.csv")
        gt = load_gt(m)
        for params in combos:
            t0 = time.time()
            res = score_labels(gt, predict(tracking, slots, **params))
            rows.append(dict(params, match=m, **res))
            print("  {} {} -> spellF1 {:.4f}  macroF1 {:.4f}  ({:.1f} s)".format(
                m, params, res["spell_f1"], res["macro_f1"], time.time() - t0))

    long = pd.DataFrame(rows)
    keys = [n for n in names]
    wide = long.pivot_table(index=keys, columns="match", values="spell_f1")
    wide.columns = ["spellF1_" + c for c in wide.columns]
    summary = long.groupby(keys).agg(snitt_spellF1=("spell_f1", "mean"), spredning=("spell_f1", "std"),
                                     min_spellF1=("spell_f1", "min"), snitt_macroF1=("macro_f1", "mean"))
    table = summary.join(wide).sort_values("snitt_spellF1", ascending=False).reset_index()

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = C.RESULTS_DIR / "sweep_{}_{}.csv".format(Path(args.grid_file).stem, stamp)
    table.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n" + table.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))
    print("\nLagret: {}".format(out))


if __name__ == "__main__":
    main()
