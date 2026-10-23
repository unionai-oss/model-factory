"""Archetype pipeline: parse, instantiate, calibrate, fit."""

import json
import random

import pytest

from resource_tuner.taskgen.archetypes import (
    Archetype,
    FootprintFit,
    calibration_points,
    instantiate,
    parse_archetype_response,
    sample_params,
)
from resource_tuner.taskgen.synthetic import RejectedTask

CODE = """\
import time
import numpy as np

PARAMS = {"rows": 1000, "cols": 8, "reps": 2}

def run() -> dict:
    deadline = time.monotonic() + 0
    total = 0
    while True:
        x = np.ones((PARAMS["rows"], PARAMS["cols"]))
        total += int(x.sum()) * PARAMS["reps"]
        del x
        if time.monotonic() >= deadline:
            break
    return {"total": total}
"""

RESPONSE = json.dumps(
    {
        "description": "matrix accumulation",
        "param_ranges": {"rows": [100, 100000], "cols": [4, 64], "reps": [1, 4]},
        "memory_param": "rows",
        "code": CODE,
    }
)


def test_parse_happy_path():
    a = parse_archetype_response(f"<think>hm</think>{RESPONSE}")
    assert a.memory_param == "rows"
    assert set(a.param_ranges) == {"rows", "cols", "reps"}


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda o: o.update(memory_param="nope"), "memory_param"),
        (lambda o: o.update(param_ranges={"rows": [100]}), "bad range"),
        (lambda o: o.update(param_ranges={"rows": [-1, 5]}), "bad range"),
        (lambda o: o["param_ranges"].update(missing=[1, 2]), "PARAMS missing"),
        (lambda o: o.update(code="def run(): return {}"), "no module-level PARAMS"),
    ],
)
def test_parse_rejections(mutate, reason):
    obj = json.loads(RESPONSE)
    mutate(obj)
    with pytest.raises(RejectedTask, match=reason):
        parse_archetype_response(json.dumps(obj))


def test_instantiate_rewrites_params_and_still_runs():
    a = parse_archetype_response(RESPONSE)
    code = instantiate(a, {"rows": 50, "cols": 4, "reps": 1})
    assert "PARAMS = {'rows': 50, 'cols': 4, 'reps': 1}" in code
    ns: dict = {}
    exec(compile(code, "variant", "exec"), ns)
    out = ns["run"]()
    assert out["total"] == 200  # 50*4*1: the sampled params really apply


def test_sample_params_respects_bounds_and_intness():
    a = parse_archetype_response(RESPONSE)
    rng = random.Random(1)
    for _ in range(50):
        p = sample_params(a, rng)
        assert 100 <= p["rows"] <= 100000 and isinstance(p["rows"], int)
        assert 4 <= p["cols"] <= 64 and isinstance(p["cols"], int)


def test_calibration_points_span_the_memory_range():
    a = parse_archetype_response(RESPONSE)
    pts = calibration_points(a, k=3)
    assert [p["rows"] for p in pts] == sorted(p["rows"] for p in pts)
    assert pts[0]["rows"] == 100 and pts[-1]["rows"] == 100000
    # non-memory params held at midpoints so the fit isolates the driver
    assert len({p["cols"] for p in pts}) == 1


def test_footprint_fit_is_linear_with_a_floor():
    fit = FootprintFit([(100.0, {"rows": 10}), (200.0, {"rows": 20})])
    assert fit.predict(30, "rows") == pytest.approx(300.0)
    assert fit.predict(15, "rows") == pytest.approx(150.0)
    # extrapolating down never collapses below the measurement floor
    assert fit.predict(0, "rows") >= 32.0
