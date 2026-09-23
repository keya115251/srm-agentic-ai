"""
Curve-shape validation for a chosen design against a target pressure-time
profile, using the exact metrics from Fan, Peng, Wei et al. (2026), Eqs.
35-36: relative impulse error and cosine similarity of the resampled
pressure curves.

Why this is a separate module rather than baked into the surrogate's
scoring function: the surrogate only predicts scalar peak values (thrust,
impulse, pressure, Kn), not full time-resolved curves, so it can't be used
to optimize toward a target curve shape directly. This module instead runs
the *real* simulator once, at the end, on whichever design the fast
scalar-based search already chose, and checks how well that design's full
curve matches a target curve you provide - e.g. a mech-team reference
motor's logged pressure-time data, or another design you want to match.

If you want the search itself to optimize for curve shape (not just
validate against it after the fact), the surrogate would need to be
extended to predict curve points, not just scalar peaks - a bigger change,
not implemented here.

Usage as a library (typical case - call after design_agent/search picks a
design):
    from design_agent.curve_matching import compare_curves
    result = compare_curves(candidate_params, target_time, target_pressure)
    print(result['cosine_similarity'], result['relative_impulse_error'])

Usage standalone (for a quick check against a CSV of target time,pressure
data):
    python design_agent/curve_matching.py --params-json '{"diameter":0.08,...}' \
        --target-csv path/to/target_curve.csv
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numpy 2.0 renamed trapz -> trapezoid; support both so this works regardless
# of which numpy version is installed.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def resample_to_grid(t_src, y_src, t_grid):
    """Linear-interpolates y_src(t_src) onto t_grid, zero-padding outside
    the source's time range (matches how a burned-out motor's pressure
    trails to zero, rather than extrapolating nonsense)."""
    return np.interp(t_grid, t_src, y_src, left=0.0, right=0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. 36 in Fan et al. (2026): normalized dot product of the two
    curves. 1.0 = identical shape, 0 = completely unrelated, can go
    negative for genuinely opposite curves (not expected here since
    pressure is always non-negative)."""
    num = np.sum(a * b)
    denom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return num / denom if denom > 0 else 0.0


def relative_impulse_error(t_a, p_a, t_b, p_b) -> float:
    """Eq. 35 in Fan et al. (2026): |I_d - I_t| / |I_t| as a percentage,
    where I is total impulse (trapezoidal integral of pressure... note the
    paper integrates chamber pressure directly as a proxy for impulse
    trend; for actual Ns-impulse comparison we integrate thrust instead if
    available - see compare_curves()."""
    I_a = _trapz(p_a, t_a)
    I_b = _trapz(p_b, t_b)
    return 100.0 * abs(I_a - I_b) / abs(I_b) if I_b != 0 else float('inf')


def compare_curves(candidate_params: dict, target_time: np.ndarray,
                    target_pressure: np.ndarray = None,
                    target_thrust: np.ndarray = None,
                    dt: float = 0.02) -> dict:
    """Runs the real simulator on candidate_params, resamples both curves
    onto a shared uniform time grid (matching the paper's approach of a
    fixed dt grid rather than comparing raw irregular timesteps), and
    reports cosine similarity + relative impulse error.

    Provide target_pressure and/or target_thrust depending on what your
    target data actually has (test-fire telemetry usually has pressure,
    a datasheet thrust curve would have thrust - the paper's Eq 35/36 use
    chamber pressure specifically, so pressure is preferred if you have it).
    """
    from simulator_bridge.bates_motor import run_bates_sim

    result = run_bates_sim(candidate_params)
    cand_time = np.array(result['time'])

    t_max = max(target_time[-1], cand_time[-1] if len(cand_time) else 0)
    t_grid = np.arange(0, t_max + dt, dt)

    out = {}

    if target_pressure is not None:
        cand_pressure = np.array(result['pressure'])
        target_p_grid = resample_to_grid(target_time, target_pressure, t_grid)
        cand_p_grid = resample_to_grid(cand_time, cand_pressure, t_grid)
        out['pressure_cosine_similarity'] = cosine_similarity(cand_p_grid, target_p_grid)
        out['pressure_relative_error_pct'] = relative_impulse_error(
            t_grid, cand_p_grid, t_grid, target_p_grid)

    if target_thrust is not None:
        cand_thrust = np.array(result['thrust'])
        target_t_grid = resample_to_grid(target_time, target_thrust, t_grid)
        cand_t_grid = resample_to_grid(cand_time, cand_thrust, t_grid)
        out['thrust_cosine_similarity'] = cosine_similarity(cand_t_grid, target_t_grid)
        # this one is genuine Ns impulse, since thrust integrates to impulse directly
        out['thrust_relative_impulse_error_pct'] = relative_impulse_error(
            t_grid, cand_t_grid, t_grid, target_t_grid)

    out['candidate_sim_result'] = result
    return out


def print_report(out: dict):
    print("=== Curve-shape validation ===")
    if 'pressure_cosine_similarity' in out:
        print(f"Pressure curve: cosine similarity={out['pressure_cosine_similarity']:.4f} "
              f"(1.0 = identical shape), relative error={out['pressure_relative_error_pct']:.2f}%")
    if 'thrust_cosine_similarity' in out:
        print(f"Thrust curve:   cosine similarity={out['thrust_cosine_similarity']:.4f} "
              f"(1.0 = identical shape), relative impulse error={out['thrust_relative_impulse_error_pct']:.2f}%")
    print("\nGuideline from the literature (Fan et al. 2026): cosine similarity")
    print("above ~0.98 with impulse error below ~2% indicates strong agreement")
    print("between the designed and target curves.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-json', required=True,
                         help='JSON dict of candidate design params, e.g. '
                              '\'{"diameter":0.08,"length":0.17,"coreDiameter":0.05,'
                              '"throat":0.014,"exit":0.079}\'')
    parser.add_argument('--target-csv', required=True,
                         help='CSV with columns: time, pressure (and/or thrust)')
    args = parser.parse_args()

    import csv
    time_vals, pressure_vals, thrust_vals = [], [], []
    with open(args.target_csv) as f:
        reader = csv.DictReader(f)
        has_pressure = 'pressure' in reader.fieldnames
        has_thrust = 'thrust' in reader.fieldnames
        for row in reader:
            time_vals.append(float(row['time']))
            if has_pressure:
                pressure_vals.append(float(row['pressure']))
            if has_thrust:
                thrust_vals.append(float(row['thrust']))

    params = json.loads(args.params_json)
    target_time = np.array(time_vals)
    target_pressure = np.array(pressure_vals) if pressure_vals else None
    target_thrust = np.array(thrust_vals) if thrust_vals else None

    out = compare_curves(params, target_time, target_pressure, target_thrust)
    print_report(out)


if __name__ == '__main__':
    main()