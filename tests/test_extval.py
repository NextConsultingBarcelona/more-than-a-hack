"""The sweep adapter gives the same answer as the file pipeline (minus Smart Tagging)."""

import numpy as np
import pandas as pd
import pytest

from src import extval, pipeline
from src.config import Config


def _frames(sim):
    names = ["s{:02d}".format(k) for k in range(sim["x"].shape[1])]
    data = {"frame_id": np.arange(1, sim["n_frames"] + 1),
            "ball_x": sim["ball"][:, 0], "ball_y": sim["ball"][:, 1]}
    for k, name in enumerate(names):
        data[name + "_x"], data[name + "_y"], data[name + "_s"] = sim["x"][:, k], sim["y"][:, k], sim["s"][:, k]
    tracking = pd.DataFrame(data)
    graded = np.zeros(sim["n_frames"], dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    tracking.loc[~graded, tracking.columns[1:]] = np.nan      # X rows have no tracking
    slots = pd.DataFrame({"slot": names, "team": np.where(sim["slot_team"] == 0, "H", "A")})
    return tracking, slots


def test_predict_matches_pipeline(fake_match):
    sim = fake_match["sim"]
    tracking, slots = _frames(sim)
    hand = dict(learned=False, c_switch=3.0, c_die=6.5, c_restart=6.0)
    # only float32 rounding of the text files differs; tree thresholds amplify it a little
    for params, agree in ((hand, 0.995), ({}, 0.97)):
        got = extval.predict(tracking, slots, pitch=(105.3, 68.0), use_tags=False, **params)
        want, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                       cfg=Config(use_tags=False).with_(**params), use_cache=False)
        assert len(got) == len(tracking)
        assert np.mean(got == want) > agree


def test_params_reach_the_model(fake_match):
    tracking, slots = _frames(fake_match["sim"])
    loose = extval.predict(tracking, slots, c_switch=1.0)
    strict = extval.predict(tracking, slots, c_switch=30.0)
    assert np.sum(loose[1:] != loose[:-1]) > np.sum(strict[1:] != strict[:-1])
    with pytest.raises(TypeError):
        extval.predict(tracking, slots, no_such_parameter=1)
