"""
Range-guarded predictions from the BATES + Finocyl hybrid surrogate. Same
interface as surrogate/finocyl_surrogate.py.

The surrogate is a tree model, so outside the ranges it was trained on its
predictions flatten out instead of following the physics. predict() checks
each design against the training ranges (raw inputs and the derived physics
features such as initial Kn) before trusting the surrogate:

    from surrogate.hybrid_surrogate import predict
    out = predict(params)                  # flag only
    out = predict(params, fallback=True)   # run the real simulator if out of range

`out` holds the four targets plus:
    in_range      True if every checked value is inside the training range
    out_of_range  {name: (value, (min, max))} for each value outside it
    source        'surrogate' or 'simulator'
"""
import os
from functools import lru_cache

import joblib
import numpy as np

from surrogate.hybrid_features import PHYSICS_FEATURES, add_physics_features

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hybrid_surrogate_model.joblib')


@lru_cache(maxsize=4)
def load_model(path: str = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(path)


def check_range(params: dict, bounds: dict, features: list) -> dict:
    """Returns {name: (value, (min, max))} for every raw input or physics
    feature of `params` that falls outside the training range."""
    x = np.array([[params[f] for f in features]], dtype=float)
    values = add_physics_features(x)[0]
    out = {}
    for name, value in zip(features + PHYSICS_FEATURES, values):
        lo, hi = bounds[name]
        if not lo <= value <= hi:
            out[name] = (float(value), (lo, hi))
    return out


def predict(params: dict, fallback: bool = False,
            model_path: str = DEFAULT_MODEL_PATH) -> dict:
    d = load_model(model_path)
    out_of_range = check_range(params, d['bounds'], d['features'])
    result = {'in_range': not out_of_range, 'out_of_range': out_of_range}

    if out_of_range and fallback:
        # Imported here so surrogate-only callers don't need motorlib.
        from simulator_bridge.hybrid_motor import run_hybrid_sim
        summary = run_hybrid_sim(params)
        result.update({t: summary[t] for t in d['targets']})
        result['source'] = 'simulator'
        return result

    x = np.array([[params[f] for f in d['features']]], dtype=float)
    result.update({t: float(m.predict(x)[0]) for t, m in d['models'].items()})
    result['source'] = 'surrogate'
    return result
