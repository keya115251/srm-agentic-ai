"""
Physics features for the BATES surrogate, kept in their own module so the
saved model pipeline can be unpickled by any caller (train_surrogate.py saves
a pipeline that calls add_physics_features on the 5 raw inputs).
"""
import numpy as np

PHYSICS_FEATURES = ['At', 'port0', 'Ab0', 'Kn0', 'AbMax', 'KnMax', 'propVol',
                    'portThroat', 'expRatio', 'web']

# Web-regression steps used to find the peak burning area analytically.
_N_STEPS = 64


def add_physics_features(X):
    """Appends geometry quantities computed from the 5 raw inputs (columns in
    FEATURES order: diameter, length, coreDiameter, throat, exit): throat
    area, initial port area, initial and peak burning area, initial and peak
    Kn, propellant volume, port/throat ratio, expansion ratio and web
    thickness. A tree model has to approximate products and ratios like
    Ab/At with many small splits; handing them over directly lifts its
    accuracy. Lives inside the saved pipeline, so callers still pass the 5
    raw inputs.

    Peak burning area follows the standard uninhibited BATES regression: after
    burning a distance x, the core is d + 2x across and the grain is L - 2x
    long, so Ab(x) = pi (d + 2x)(L - 2x) + 2 * pi/4 (D^2 - (d + 2x)^2).
    """
    X = np.asarray(X, dtype=float)
    diameter, length, core, throat, exit_dia = X.T
    at = np.pi / 4 * throat ** 2
    port0 = np.pi / 4 * core ** 2
    web = (diameter - core) / 2

    x = web[:, None] * np.linspace(0, 1, _N_STEPS)[None, :]
    d_x = core[:, None] + 2 * x
    l_x = np.maximum(length[:, None] - 2 * x, 0)
    ab = np.pi * d_x * l_x + 2 * np.pi / 4 * (diameter[:, None] ** 2 - d_x ** 2)
    ab0 = ab[:, 0]
    ab_max = ab.max(axis=1)

    extra = np.column_stack([
        at, port0, ab0, ab0 / at, ab_max, ab_max / at,
        np.pi / 4 * (diameter ** 2 - core ** 2) * length,
        port0 / at, (exit_dia / throat) ** 2, web,
    ])
    return np.hstack([X, extra])
