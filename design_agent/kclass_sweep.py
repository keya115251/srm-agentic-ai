"""
Sweeps the hybrid optimizer (DE + Nelder-Mead) across several target-impulse
values spanning the K-class range (1280.01-2560.00 Ns), rather than
optimizing for a single midpoint value. For each target, validates the
resulting design against a reference curve (e.g. the real K950) using
curve_matching.py's cosine similarity, and reports which impulse target
produces the best shape match.

Why this can find a better match than a single midpoint run: the surrogate
optimizes for impulse only, so different impulse targets land on genuinely
different grain geometries, which produce different curve *shapes* as a
side effect. This sweep checks that empirically instead of assuming the
midpoint is best.

This does NOT optimize for curve shape directly (the surrogate can't, it
only predicts scalar peaks). It is a coarse search over the one dimension
we can cheaply vary (target impulse).

Changes from v1:
  - EXIT_THROAT_RATIO_BOUNDS narrowed from (3.0, 9.0) to (1.5, 3.5), matching
    the corrected training data. After the ambPressure fix, most of the old
    range causes flow separation at real sea-level pressure, and the
    surrogate is no longer trained there.
  - K-class validity is now checked against the REAL simulated impulse, not
    just the target. The surrogate can drift (up to ~13% observed on
    optimized designs), so a design targeted inside K-class can come out
    as L-class in the real simulator. Such designs are marked INVALID and
    excluded from "best match" selection.
  - Reports surrogate drift (surrogate-predicted vs real impulse) per row,
    so optimizer-induced extrapolation into sparse regions is visible.
  - Suppresses motorlib's known-harmless nozzle.py RuntimeWarnings (fsolve
    noise on candidate designs), which previously buried the output.

Usage:
    python design_agent/kclass_sweep.py --target-csv data/k950_target_curve.csv
"""
import argparse
import csv
import os
import sys
import warnings

import joblib
import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_agent.curve_matching import compare_curves

warnings.filterwarnings('ignore', category=RuntimeWarning, module='motorlib')

SURROGATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'surrogate', 'surrogate_model.joblib'
)

# Must match the bounds used to generate the surrogate's training data
# (scripts/generate_sweep_data_random.py). Searching outside them means the
# surrogate is extrapolating and its predictions are unreliable.
DIAMETER_BOUNDS = (0.06, 0.10)
LENGTH_BOUNDS = (0.10, 0.18)
CORE_DIAMETER_FRAC_BOUNDS = (0.30, 0.75)
THROAT_BOUNDS = (0.010, 0.018)
EXIT_THROAT_RATIO_BOUNDS = (1.5, 3.5)

BOUNDS = [DIAMETER_BOUNDS, LENGTH_BOUNDS, CORE_DIAMETER_FRAC_BOUNDS,
          THROAT_BOUNDS, EXIT_THROAT_RATIO_BOUNDS]
FEATURE_ORDER = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']

K_CLASS_MIN = 1280.01
K_CLASS_MAX = 2560.00


def decode_params(x):
    diameter, length, core_frac, throat, exit_ratio = x
    return {'diameter': float(diameter), 'length': float(length),
            'coreDiameter': float(diameter * core_frac), 'throat': float(throat),
            'exit': float(throat * exit_ratio)}


def make_objective(models, target_impulse, max_pressure=None):
    def objective(x):
        params = decode_params(x)
        X = np.array([[params[f] for f in FEATURE_ORDER]])
        impulse = models['total_impulse'].predict(X)[0]
        pressure = models['peak_pressure'].predict(X)[0]
        score = abs(impulse - target_impulse) / target_impulse
        if max_pressure is not None and pressure > max_pressure:
            score += 5.0 * (pressure - max_pressure) / max_pressure
        return score
    return objective


def optimize_for_target(models, target_impulse, max_pressure, seed=0):
    """Same DE + Nelder-Mead hybrid as search_v3_hybrid.py, for one target."""
    objective = make_objective(models, target_impulse, max_pressure)
    de_result = differential_evolution(objective, BOUNDS, popsize=20,
                                        maxiter=100, seed=seed, tol=1e-6,
                                        polish=False)
    nm_result = minimize(objective, x0=de_result.x, method='Nelder-Mead',
                          bounds=BOUNDS,
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    best_x = nm_result.x if nm_result.fun <= de_result.fun else de_result.x
    return decode_params(best_x)


def surrogate_impulse(models, params):
    X = np.array([[params[f] for f in FEATURE_ORDER]])
    return float(models['total_impulse'].predict(X)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-csv', required=True,
                         help='CSV with the reference curve (columns: time, thrust)')
    parser.add_argument('--max-pressure', type=float, default=6e6)
    parser.add_argument('--n-points', type=int, default=9,
                         help='number of impulse targets across the K-class range')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SURROGATE_PATH):
        print(f"ERROR: surrogate model not found at {SURROGATE_PATH}.")
        return

    models = joblib.load(SURROGATE_PATH)['models']

    time_vals, thrust_vals = [], []
    with open(args.target_csv) as f:
        for row in csv.DictReader(f):
            time_vals.append(float(row['time']))
            thrust_vals.append(float(row['thrust']))
    target_time = np.array(time_vals)
    target_thrust = np.array(thrust_vals)

    impulse_targets = np.linspace(K_CLASS_MIN, K_CLASS_MAX, args.n_points)

    print(f"Sweeping {args.n_points} impulse targets across the K-class range "
          f"({K_CLASS_MIN:.0f}-{K_CLASS_MAX:.0f} Ns)...\n")

    results = []
    for i, target_impulse in enumerate(impulse_targets, 1):
        print(f"[{i}/{len(impulse_targets)}] target={target_impulse:.0f} Ns ... ",
              end='', flush=True)
        params = optimize_for_target(models, target_impulse, args.max_pressure,
                                      seed=args.seed)
        pred_impulse = surrogate_impulse(models, params)

        try:
            match = compare_curves(params, target_time, target_thrust=target_thrust)
        except Exception as e:
            print(f"FAILED - {e}")
            continue

        sim = match['candidate_sim_result']
        real_impulse = sim['total_impulse']
        drift = abs(real_impulse - pred_impulse) / real_impulse
        valid = K_CLASS_MIN <= real_impulse <= K_CLASS_MAX
        row = {
            'target_impulse': target_impulse,
            'params': params,
            'cosine_similarity': match['thrust_cosine_similarity'],
            'impulse_error_vs_ref_pct': match['thrust_relative_impulse_error_pct'],
            'pred_impulse': pred_impulse,
            'real_impulse': real_impulse,
            'drift': drift,
            'valid': valid,
            'alerts': sim['alerts'],
        }
        results.append(row)
        print(f"cosine_sim={row['cosine_similarity']:.4f}  "
              f"real={real_impulse:.0f}Ns  drift={drift:.1%}  "
              f"{'OK' if valid else 'INVALID (outside K-class)'}")

    if not results:
        print("\nNo successful runs - check that motorlib is importable (PYTHONPATH set).")
        return

    valid_results = [r for r in results if r['valid']]
    invalid_results = [r for r in results if not r['valid']]

    print(f"\n=== Full ranking (by cosine similarity) ===")
    print(f"  {'target':>7} {'real':>7} {'drift':>7} {'cos_sim':>8} {'status':>8}")
    for r in sorted(results, key=lambda r: -r['cosine_similarity']):
        print(f"  {r['target_impulse']:>7.0f} {r['real_impulse']:>7.0f} "
              f"{r['drift']:>7.1%} {r['cosine_similarity']:>8.4f} "
              f"{'OK' if r['valid'] else 'INVALID':>8}")

    if invalid_results:
        print(f"\n{len(invalid_results)} design(s) excluded: real impulse outside "
              f"K-class ({K_CLASS_MIN:.2f}-{K_CLASS_MAX:.2f} Ns) despite an in-range "
              f"target, due to surrogate drift.")

    if not valid_results:
        print("\nNo valid K-class designs found.")
        return

    best = max(valid_results, key=lambda r: r['cosine_similarity'])
    drifts = [r['drift'] for r in results]
    print(f"\n=== Best valid K-class match ===")
    print(f"Target impulse: {best['target_impulse']:.0f} Ns "
          f"(surrogate: {best['pred_impulse']:.0f} Ns, real: {best['real_impulse']:.0f} Ns, "
          f"drift {best['drift']:.1%})")
    print(f"Cosine similarity vs reference: {best['cosine_similarity']:.4f}")
    print(f"Impulse error vs reference: {best['impulse_error_vs_ref_pct']:.2f}% "
          f"(informational only; KNSB vs APCP means absolute impulse is not a target)")
    print(f"Alerts: {best['alerts'] or 'none'}")
    print(f"Params: {best['params']}")
    print(f"\nSurrogate drift across sweep: mean {np.mean(drifts):.1%}, "
          f"max {np.max(drifts):.1%}")


if __name__ == '__main__':
    main()