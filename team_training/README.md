# Training of the second home-vs-away model

`src/team_model.py` and the stored trees in `models/team_trees.npz` are all the competition
pipeline needs. This folder holds the code that trained and validated them. Nothing here is
needed to produce the submission, and nothing here reads the competition match.

## What it is trained on

The five tuning matches of the Sportec/IDSSE open data (Bassek et al. 2025, CC-BY 4.0), in the
form `external_validation/build_dataset.py` gives them, each in three pictures of the live ball
(`pictures.py`): as built, with the ball missing while it is in the air, and with position
noise. The two held-out matches are not used.

The label is Sportec's ownership field with uncontrolled touches removed: a team spell shorter
than 2 s between two spells of the other team belongs to the other team. `ownership.py` shows
why: the field changes whenever the other team touches the ball, and the organiser counts only
controlled possession.

## Steps

```bash
python -m pip install -r external_validation/requirements-validation.txt
python external_validation/build_dataset.py --matches all   # the team's validation data
python dead_training/pack_sportec.py                        # the clean matches, for the ball's height
python team_training/ownership.py                           # what the ownership field means
python team_training/train.py                               # about 7 minutes on one core
```

`train.py` writes `models/team_trees.npz` (trained on the five tuning matches) and
`ext_data/team_lomo/<match>.npz`, one model per tuning match trained on the other four.

## Validation

```bash
python dead_training/train.py --lomo                        # leave-one-match-out stoppage detectors
python team_training/compare.py --jobs 2                    # the table of NEXT.md 5.9, about 20 min
python team_training/compare.py --live --jobs 2             # other pictures of the live ball
```

`compare.py` labels every match with three versions of the pipeline (the model first tagged
`phase1-submission`, that model with the stoppage detector, and the default with both the
detector and the second home-vs-away model), each with models that have not seen the match,
and scores them against Sportec's ownership as it is and with uncontrolled touches removed.

A quicker look through the team's own sweep on the validation data as built:

```bash
python external_validation/sweep.py external_validation/grids/team_model.json
```

Here the first model (`src/learned_model.json`) is the stored one for every match, so only the
two held-out matches (J03WQQ, J03WR9) are unseen by all models.
