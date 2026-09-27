"""
Physics features for the Finocyl surrogate, kept in their own module so the
saved model pipeline can be unpickled by any caller (train_finocyl_surrogate.py
saves a pipeline that calls add_physics_features on the 8 raw inputs).
"""
import numpy as np

PHYSICS_FEATURES = ['At', 'port0', 'perim0', 'endArea', 'Ab0', 'Kn0', 'propVol',
                    'portThroat', 'expRatio', 'finWeb', 'coreWeb', 'finArea']


def add_physics_features(X):
    """Appends initial-geometry quantities computed from the 8 raw inputs
    (columns in FEATURES order): throat area, initial port area, burning
    perimeter and burning area, initial Kn, propellant volume, port/throat
    ratio, expansion ratio, web thicknesses and fin side area. A tree model
    has to approximate products and ratios like Ab0/At with many small
    splits; handing them over directly is what lifts CV R^2 from ~0.93 to
    ~0.99. Lives inside the saved pipeline, so callers still pass the 8 raw
    inputs.
    """
    X = np.asarray(X, dtype=float)
    diameter, length, core, num_fins, fin_w, fin_l, throat, exit_dia = X.T
    rc = core / 2
    at = np.pi / 4 * throat ** 2
    port0 = np.pi * rc ** 2 + num_fins * fin_w * fin_l
    perim0 = 2 * np.pi * rc - num_fins * fin_w + num_fins * (2 * fin_l + fin_w)
    end_area = np.pi / 4 * diameter ** 2 - port0
    ab0 = perim0 * length + 2 * end_area   # ends uninhibited, as in the wrapper
    extra = np.column_stack([
        at, port0, perim0, end_area, ab0, ab0 / at, end_area * length,
        port0 / at, (exit_dia / throat) ** 2, diameter / 2 - rc - fin_l,
        diameter / 2 - rc, num_fins * 2 * fin_l * length,
    ])
    return np.hstack([X, extra])
