"""Tren de lærte evidensmodellene (src/learned.py) på åpne Sportec-kamper og evaluer dem.

Bare åpne data: leser ext_data/<kamp>/ (bygget av build_dataset.py), aldri konkurransedata.

Hva skriptet gjør:
  1. Regner ut trekkene for hver kamp med nøyaktig samme kode som pipelinen bruker (src/).
  2. Leave-one-match-out på tuningkampene: tren på fire, prediker den femte. Slik er ingen
     tuning-score regnet på en kamp modellen har sett.
  3. Søker dekoderverdier (skala på evidensen og overgangskostnader) på disse prediksjonene og
     velger kombinasjonen med høyest snitt-spellF1.
  4. Trener endelig modell på alle tuningkampene, scorer de to holdt-utenfor-kampene én gang
     med den valgte kombinasjonen, og skriver modellen til src/learned_model.json.

Kjør fra repo-mappen (krever lightgbm, se requirements-validation.txt):
  python external_validation/train_emissions.py
  python external_validation/train_emissions.py --quick      # bare standardverdiene, ingen søk

Alt er deterministisk: fast seed, én tråd, ingen tilfeldig radutvelgelse.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from score import load_gt, score_labels  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT))
from src import deadball, extval, features, learned, possession  # noqa: E402
from src.config import Config  # noqa: E402

TUNING = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY"]
HELD_OUT = ["J03WQQ", "J03WR9"]
STRIDE = 5          # naborammer er nesten kopier: tren på hver femte
LGB_PARAMS = dict(objective="binary", num_leaves=15, learning_rate=0.06, min_data_in_leaf=200,
                  feature_fraction=0.8, lambda_l2=1.0, use_missing=False, deterministic=True,
                  force_row_wise=True, num_threads=1, seed=0, verbose=-1)
GRID = dict(learned_team_scale=[0.05, 0.1, 0.2, 0.4], learned_dead_scale=[0.05, 0.1, 0.2, 0.4],
            c_switch=[2.0, 3.0, 4.0, 6.0], c_stop=[4.0, 6.5, 10.0])


def prepare(match):
    """Trekk, fasit og det dekoderen trenger for én kamp."""
    tracking = pd.read_parquet(C.OUT_DIR / match / "tracking.parquet")
    slots = pd.read_csv(C.OUT_DIR / match / "slots.csv")
    trk, blocks = extval._tracking(tracking, slots)
    cfg = Config()
    feats = features.compute(trk, extval.PITCH, cfg)
    dead_hand, _ = deadball.compute(feats, [], blocks, cfg)
    ball_llr, ctrl = possession.ball_evidence(feats, cfg)
    hint = possession.hint_evidence(feats, ball_llr, dead_hand, blocks, cfg)
    anti, sym = learned.team_features(feats, ball_llr, hint, dead_hand, blocks)
    gt = load_gt(match)
    graded = np.zeros(feats.n, dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    extval._cache.clear()
    return dict(match=match, feats=feats, ctrl=ctrl, blocks=blocks, anti=anti, sym=sym,
                dead_x=learned.dead_features(feats, ball_llr, dead_hand), gt=gt, graded=graded,
                labels=gt["label"].astype(str).str.strip().str.upper().to_numpy())


def to_arrays(booster, prior_logit=0.0):
    """LightGBM-modellen som rene tabeller (src/learned.predict_trees leser dem)."""
    trees = []
    for info in booster.dump_model()["tree_info"]:
        f, t, left, right, v = [], [], [], [], []

        def walk(node, depth):
            i = len(f)
            for arr in (f, t, left, right, v):
                arr.append(0)
            if "leaf_value" in node and "split_feature" not in node:
                f[i], t[i], left[i], right[i], v[i] = -1, 0.0, i, i, float(node["leaf_value"])
                return i, depth
            assert node["decision_type"] == "<=", node["decision_type"]
            f[i], t[i] = int(node["split_feature"]), float(node["threshold"])
            left[i], d1 = walk(node["left_child"], depth + 1)
            right[i], d2 = walk(node["right_child"], depth + 1)
            return i, max(d1, d2)

        _, depth = walk(info["tree_structure"], 0)
        trees.append(dict(f=f, t=t, l=left, r=right, v=v, depth=depth))
    return dict(trees=trees, prior_logit=float(prior_logit))


def fit(matches, n_trees):
    """Tren begge modellene på en liste forberedte kamper."""
    import lightgbm as lgb

    dx, dy, tx, ty = [], [], [], []
    for m in matches:
        rows = np.flatnonzero(m["graded"] & (m["labels"] != "X"))[::STRIDE]
        dx.append(m["dead_x"][rows])
        dy.append((m["labels"][rows] == "D").astype(float))
        rows = np.flatnonzero(np.isin(m["labels"], ["H", "A"]))[::STRIDE]
        home = (m["labels"][rows] == "H").astype(float)
        # begge orienteringer: byttes hjemme og borte, snur de antisymmetriske trekkene fortegn
        tx += [np.column_stack([m["anti"][rows], m["sym"][rows]]),
               np.column_stack([-m["anti"][rows], m["sym"][rows]])]
        ty += [home, 1.0 - home]
    dx, dy, tx, ty = np.vstack(dx), np.concatenate(dy), np.vstack(tx), np.concatenate(ty)
    dead = lgb.train(LGB_PARAMS, lgb.Dataset(dx, dy), num_boost_round=n_trees)
    team = lgb.train(LGB_PARAMS, lgb.Dataset(tx, ty), num_boost_round=n_trees)
    share = float(dy.mean())
    model = dict(dead=to_arrays(dead, np.log(share / (1.0 - share))), team=to_arrays(team),
                 n_dead_features=int(dx.shape[1]), n_team_features=int(tx.shape[1]))
    for name, booster, x in (("dead", dead, dx[:2000]), ("team", team, tx[:2000])):
        diff = np.abs(learned.predict_trees(model[name], x) - booster.predict(x, raw_score=True))
        assert diff.max() < 1e-9, "numpy-trærne avviker fra LightGBM: {}".format(diff.max())
    return model


def logits(model, m):
    full = np.column_stack([m["anti"], m["sym"]])
    mirrored = np.column_stack([-m["anti"], m["sym"]])
    team = 0.5 * (learned.predict_trees(model["team"], full)
                  - learned.predict_trees(model["team"], mirrored))
    dead = learned.predict_trees(model["dead"], m["dead_x"]) - model["dead"]["prior_logit"]
    return team, dead


def score(m, z, params):
    cfg = Config().with_(learned=True, learned_team_scale=params["learned_team_scale"],
                         learned_dead_scale=params["learned_dead_scale"],
                         c_switch=params["c_switch"], c_die=params["c_stop"],
                         c_restart=params["c_stop"])
    zero = np.zeros(m["feats"].n)
    team, dead = possession.learned_llr(z[0], z[1], zero, zero, m["graded"], cfg)
    states = possession.decode_states(team, dead, m["ctrl"], m["feats"], m["blocks"], None, cfg)
    return score_labels(m["gt"], possession._labels(states, m["blocks"], m["feats"].n))


_WORK = {}


def _row(params):
    prepared, zs = _WORK["prepared"], _WORK["zs"]
    res = [score(m, zs[m["match"]], params) for m in prepared]
    spell = [r["spell_f1"] for r in res]
    return dict(params, snitt_spellF1=np.mean(spell), spredning=np.std(spell, ddof=1),
                min_spellF1=min(spell), snitt_macroF1=np.mean([r["macro_f1"] for r in res]),
                **{"spellF1_" + m["match"]: s for m, s in zip(prepared, spell)})


def table(prepared, zs, combos, jobs=1):
    """Én rad per kombinasjon. jobs > 1 bruker flere prosesser der 'fork' finnes (ikke Windows);
    resultatet er det samme uansett."""
    import multiprocessing as mp

    _WORK.update(prepared=prepared, zs=zs)
    if jobs > 1 and len(combos) > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(jobs) as pool:
            rows = pool.map(_row, combos)
    else:
        rows = [_row(params) for params in combos]
    return pd.DataFrame(rows).sort_values("snitt_spellF1", ascending=False, kind="stable")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="bare standardverdiene, ingen søk")
    ap.add_argument("--jobs", type=int, default=1, help="prosesser i søket (påvirker ikke resultatet)")
    ap.add_argument("--out", default=str(learned.MODEL_PATH))
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    fmt = dict(index=False, float_format=lambda v: "{:.4f}".format(v))

    t0 = time.time()
    tuning = [prepare(m) for m in TUNING]
    held = [prepare(m) for m in HELD_OUT]
    print("trekk for {} kamper: {:.0f} s".format(len(tuning) + len(held), time.time() - t0))

    zs = {}
    for m in tuning:
        model = fit([o for o in tuning if o is not m], args.trees)
        zs[m["match"]] = logits(model, m)
    print("leave-one-match-out ferdig: {:.0f} s".format(time.time() - t0))

    names = list(GRID)
    combos = [dict(zip(names, v)) for v in itertools.product(*(GRID[n] for n in names))]
    if args.quick:
        combos = [dict(learned_team_scale=0.2, learned_dead_scale=0.2, c_switch=3.0, c_stop=6.5)]
    tune = table(tuning, zs, combos, args.jobs)
    print("\nTuningkamper (leave-one-match-out):\n" + tune.head(25).to_string(**fmt))
    best = {n: float(tune.iloc[0][n]) for n in names}

    final = fit(tuning, args.trees)
    held_zs = {m["match"]: logits(final, m) for m in held}
    print("\nHoldt utenfor, valgt kombinasjon {}:\n".format(best)
          + table(held, held_zs, [best]).to_string(**fmt))

    final["decoder"] = best
    final["trained_on"] = TUNING
    final["lgb_params"] = LGB_PARAMS
    final["n_trees"] = args.trees
    Path(args.out).write_text(json.dumps(final, separators=(",", ":")), encoding="utf-8")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tune.to_csv(C.RESULTS_DIR / "learned_tuning_{}.csv".format(stamp), index=False)
    print("\nModell: {} ({:.0f} kB)   total tid {:.0f} s".format(
        args.out, Path(args.out).stat().st_size / 1e3, time.time() - t0))


if __name__ == "__main__":
    main()
