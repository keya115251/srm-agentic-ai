"""
Checks of design_agent/curve_matching.thrust_match() on synthetic curves: the
absolute errors it reports, and that it separates curves cosine similarity
cannot (same shape at a different thrust level). Run with: python -m pytest test/
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from design_agent.curve_matching import burn_window, thrust_match


def trapezoid_curve(level=1000.0, burn=2.0, ramp=0.05):
    """Ignition at t=0, linear rise and fall of `ramp` s, flat at `level`."""
    t = np.array([0.0, ramp, burn - ramp, burn])
    return t, np.array([0.0, level, level, 0.0])


def test_identical_curves_score_zero():
    t, f = trapezoid_curve()
    m = thrust_match(t, f, t, f)
    assert m['match_error'] == pytest.approx(0.0, abs=1e-9)
    assert m['cosine_similarity'] == pytest.approx(1.0)


def test_half_thrust_is_caught_where_cosine_is_not():
    t, f = trapezoid_curve()
    m = thrust_match(t, f / 2, t, f)
    assert m['cosine_similarity'] == pytest.approx(1.0)
    assert m['impulse_error'] == pytest.approx(-0.5, rel=1e-3)
    assert m['avg_thrust_error'] == pytest.approx(-0.5, rel=1e-3)
    assert m['burn_time_error'] == pytest.approx(0.0, abs=1e-2)
    assert m['match_error'] > 0.5


def test_burn_time_and_impulse_errors_for_a_longer_burn():
    t, f = trapezoid_curve(burn=2.0)
    t_long, f_long = trapezoid_curve(burn=2.5)
    m = thrust_match(t_long, f_long, t, f)
    assert m['burn_time_error'] == pytest.approx(0.25, rel=0.02)
    # 0.5 s more at full thrust on a 1.95 Ns/N target.
    assert m['impulse_error'] == pytest.approx(0.5 / 1.95, rel=0.01)
    assert m['avg_thrust_error'] == pytest.approx(0.0, abs=0.01)


def test_nrmse_is_scale_free():
    """Scaling both curves' thrust by the same factor leaves every error unchanged,
    so the score means the same for a G motor and an M motor."""
    t, f = trapezoid_curve()
    t2, f2 = trapezoid_curve(level=800.0, burn=1.8)
    a = thrust_match(t2, f2, t, f)
    b = thrust_match(t2, 7 * f2, t, 7 * f)
    for k in ('nrmse', 'impulse_error', 'burn_time_error', 'match_error'):
        assert a[k] == pytest.approx(b[k], rel=1e-9)


def test_closer_level_scores_better():
    t, f = trapezoid_curve()
    errs = [thrust_match(t, s * f, t, f)['match_error'] for s in (0.5, 0.8, 0.95, 1.1, 1.5)]
    assert errs[2] < errs[1] < errs[0]
    assert errs[2] < errs[3] < errs[4]


def test_burn_window_threshold():
    t = np.linspace(0, 1, 101)
    y = np.where(t < 0.5, 100.0, 4.0)  # tail below 5% of peak is not burn
    assert burn_window(t, y) == (0.0, pytest.approx(0.49))


def test_match_curve_encoders_invert_the_decoders():
    """match_curve.py refines library designs by encoding their params back
    into search_multigrain's optimizer vectors; decoding must give them back."""
    pytest.importorskip('motorlib')
    import design_agent.search_multigrain as sm
    from design_agent.match_curve import ENCODERS
    rng = np.random.default_rng(0)
    for name, grain in sm.GRAINS.items():
        lo, hi = np.array(grain.bounds).T
        for _ in range(20):
            params = grain.decode(lo + rng.random(len(lo)) * (hi - lo))
            again = grain.decode(np.array(ENCODERS[name](params)))
            assert again.keys() == params.keys()
            for k in params:
                assert again[k] == pytest.approx(params[k], rel=1e-9), (name, k)
