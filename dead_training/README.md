# dead_training – how the dead-ball detector was made

`src/dead_model.py` reads a trained tree ensemble from `models/dead_trees.npz` and needs numpy
only. This folder is the code that produced that file. **It is not needed to reproduce the
submission**, and it never reads the competition match.

## Data

Seven matches of the Sportec/IDSSE open tracking dataset (Bundesliga 2022/23, CC-BY 4.0;
Bassek et al. 2025, *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y): J03WMX, J03WN1,
J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9. The label is Sportec's own per-frame ball status
(dead / alive). We labelled nothing ourselves.

## Steps

```bash
python -m pip install -r dead_training/requirements-training.txt
python dead_training/pack_sportec.py     # download with kloppy, pack to ext_data/dead_training/sportec/
python dead_training/train.py            # -> models/dead_trees.npz
python dead_training/train.py --lomo     # also seven leave-one-match-out detectors, for validation
python dead_training/check_features.py   # run-time features == training-time features
```

1. **Pack** (`pack_sportec.py`). One compact file per match: positions of all players and the
   ball, ball status, ball-owning team, period.
2. **Degrade** (`deadball/degrade.py`). The open matches are corrected tracking; automatic tracking
   is not. Each match is copied twice with different random seeds, and in every copy the ball is
   removed on roughly 30–50 % of the live frames (more in crowds and when it is in the air),
   player tracks get gaps and correlated position noise, and false ball detections are inserted.
3. **The ball during stoppages.** Sportec freezes the ball where play stopped and moves it to the
   restart point on the restart frame. That is a placeholder, not an observation, so it is
   removed. Each stoppage then gets one of several pictures at random: no ball; a ball that
   rolls on for a moment; a ball lying at the restart point; a ball moved to the restart point;
   a ball with the player who takes the restart. The detector therefore cannot lean on any one
   of them.
4. **Features** (`deadball/features.py`). 279 per frame; the same definitions as the run-time
   code in `src/dead_model.py`, which is a numpy-only rewrite of this file.
5. **Fit** (`deadball/model.py`). scikit-learn `HistGradientBoostingClassifier`, 300 trees of up
   to 63 leaves, learning rate 0.08, fixed seed. Training rows: every third in-play frame plus
   every frame within 2 s of a dead/alive boundary.
6. **Export** (`export_trees.py`). The trees as flat numpy arrays. `src/dead_model.raw_scores`
   walks them and returns exactly the numbers scikit-learn's `decision_function` returns
   (checked: zero difference on 142 345 frames).

## Validation

The stored detector has seen all seven matches, so it cannot be scored on them. The scores in
`NEXT.md` (section 5.8) are leave-one-match-out: `train.py --lomo` writes seven detectors,
each trained on six matches, and every validation match is scored with the detector that has
not seen it. `stoppage_pictures.py` produces the table of section 5.8: the pipeline with and
without the detector, on the validation data as built and with other pictures of the ball
during stoppages, with the tree models of `src/learned.py` refitted leave-one-match-out as well:

```bash
python external_validation/build_dataset.py --matches all   # the team's validation data
python dead_training/stoppage_pictures.py --jobs 2          # about 20 minutes on two cores
```

A quicker check on the validation data as built, through the team's own sweep
(`src.extval:predict_lomo`). Here the stored `src/learned_model.json` is used for every match,
so only the two held-out matches (J03WQQ, J03WR9) are unseen by all models:

```bash
python external_validation/sweep.py external_validation/grids/dead_model.json
```

## Reproducibility

Seeds are fixed everywhere. Gradient boosting adds floating-point numbers in parallel, so a
retrained detector is equivalent to the stored one but not guaranteed to be bit-identical. The
submission is reproduced from the stored file.
