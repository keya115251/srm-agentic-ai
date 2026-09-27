"""
Physics features for the BATES + Finocyl hybrid surrogate, kept in their own
module so the saved model pipeline can be unpickled by any caller
(train_hybrid_surrogate.py saves a pipeline that calls add_physics_features
on the 10 raw inputs). Same idea as surrogate/finocyl_features.py, extended
to the two-segment motor.
"""
import numpy as np

PHYSICS_FEATURES = ['At', 'finPort0', 'finAb0', 'batesPort0', 'batesAb0',
                    'Ab0', 'Kn0', 'propVol', 'finPortThroat', 'batesPortThroat',
                    'fluxProxy', 'expRatio', 'finWeb', 'finCoreWeb', 'batesWeb']


def add_physics_features(X):
    """Appends initial-geometry quantities computed from the 10 raw inputs
    (columns in train_hybrid_surrogate.FEATURES order): throat area, each
    segment's initial port and burning area, total burning area and initial
    Kn, propellant volume, each segment's port/throat ratio, an aft mass-flux
    proxy (total burning area over the BATES port, which all gas passes
    through), expansion ratio, and web thicknesses. A tree model has to
    approximate products and ratios like Ab0/At with many small splits;
    handing them over directly is what lifts accuracy. Lives inside the
    saved pipeline, so callers still pass the 10 raw inputs.
    """
    X = np.asarray(X, dtype=float)
    (diameter, fin_len_seg, fin_core, num_fins, fin_w, fin_l,
     bates_len, bates_core, throat, exit_dia) = X.T
    case_area = np.pi / 4 * diameter ** 2
    at = np.pi / 4 * throat ** 2

    # Finocyl segment, ends uninhibited as in the wrapper
    rc = fin_core / 2
    fin_port0 = np.pi * rc ** 2 + num_fins * fin_w * fin_l
    fin_perim0 = 2 * np.pi * rc - num_fins * fin_w + num_fins * (2 * fin_l + fin_w)
    fin_end = case_area - fin_port0
    fin_ab0 = fin_perim0 * fin_len_seg + 2 * fin_end

    # BATES segment, ends uninhibited
    bates_port0 = np.pi / 4 * bates_core ** 2
    bates_end = case_area - bates_port0
    bates_ab0 = np.pi * bates_core * bates_len + 2 * bates_end

    ab0 = fin_ab0 + bates_ab0
    extra = np.column_stack([
        at, fin_port0, fin_ab0, bates_port0, bates_ab0,
        ab0, ab0 / at, fin_end * fin_len_seg + bates_end * bates_len,
        fin_port0 / at, bates_port0 / at, ab0 / bates_port0,
        (exit_dia / throat) ** 2,
        diameter / 2 - rc - fin_l, diameter / 2 - rc,
        np.minimum((diameter - bates_core) / 2, bates_len / 2),
    ])
    return np.hstack([X, extra])
