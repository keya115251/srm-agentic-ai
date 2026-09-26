"""
Sweeps the hybrid optimizer (DE + Nelder-Mead) across several target-impulse
values spanning the K-class range (1280.01-2560.00 Ns), rather than
optimizing for a single midpoint value. For each target, validates the
resulting design against the real K950 curve using curve_matching.py's
cosine similarity, and reports which impulse target actually produces the
best shape match.

Why this can find a better match than a single midpoint run: the surrogate
optimizes for impulse only, so different impulse targets land on genuinely
different grain geometries, which produce different curve *shapes* as a
side effect. There's no guarantee 1920 Ns (the midpoint) is the target
whose resulting shape happens to look most like K950 - this sweep checks
that empirically instead of assuming it.

This does NOT optimize for curve shape directly (the surrogate can't, it
only predicts scalar peaks) - it's a coarse search over the one dimension
we can cheaply vary (target impulse) to see if any point in the valid
K-class range does noticeably better than the midpoint by coincidence of
the underlying physics.

Usage:
    python design_agent/kclass_sweep.py --target-csv data/k950_target_curve.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_agent.curve_matching import compare_curves

SURROGATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'surrogate', 'surrogate_model.joblib'
)

DIAMETER_BOUNDS = (0.06, 0.10)
LENGTH_BOUNDS = (0.10, 0.18)
CORE_DIAMETER_FRAC_BOUNDS = (0.30, 0.75)
THROAT_BOUNDS = (0.010, 0.018)
EXIT_THROAT_RATIO_BOUNDS = (3.0, 9.0)

BOUNDS = [DIAMETER_BOUNDS, LENGTH_BOUNDS, CORE_DIAMETER_FRAC_BOUNDS,
          THROAT_BOUNDS, EXIT_THROAT_RATIO_BOUNDS]
FEATURE_ORDER = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']

K_CLASS_MIN = 1280.01
K_CLASS_MAX = 2560.00


def decode_params(x):
    diameter, length, core_frac, throat, exit_ratio = x
    return {'diameter': diameter, 'length': length,
            'coreDiameter': diameter * core_frac, 'throat': throat,
            'exit': throat * exit_ratio}


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
    """Runs the same DE + Nelder-Mead hybrid as search_v3_hybrid.py for one
    target-impulse value, returns the best params found."""
    objective = make_objective(models, target_impulse, max_pressure)
    de_result = differential_evolution(objective, BOUNDS, popsize=20,
                                        maxiter=100, seed=seed, tol=1e-6,
                                        polish=False)
    nm_result = minimize(objective, x0=de_result.x, method='Nelder-Mead',
                          bounds=BOUNDS,
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    best_x = nm_result.x if nm_result.fun <= de_result.fun else de_result.x
    return decode_params(best_x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-csv', required=True,
                         help='CSV with the real K950 (or other reference) curve')
    parser.add_argument('--max-pressure', type=float, default=6e6)
    parser.add_argument('--n-points', type=int, default=9,
                         help='number of impulse targets to try across the K-class range')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SURROGATE_PATH):
        print(f"ERROR: surrogate model not found at {SURROGATE_PATH}.")
        return

    d = joblib.load(SURROGATE_PATH)
    models = d['models']

    import csv
    time_vals, thrust_vals = [], []
    with open(args.target_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_vals.append(float(row['time']))
            thrust_vals.append(float(row['thrust']))
    target_time = np.array(time_vals)
    target_thrust = np.array(thrust_vals)

    impulse_targets = np.linspace(K_CLASS_MIN, K_CLASS_MAX, args.n_points)

    print(f"Sweeping {args.n_points} impulse targets across the K-class range "
          f"({K_CLASS_MIN:.0f}-{K_CLASS_MAX:.0f} Ns)...\n")

    results = []
    for i, target_impulse in enumerate(impulse_targets, 1):
        print(f"[{i}/{len(impulse_targets)}] target_impulse={target_impulse:.1f} Ns ... ", end='', flush=True)
        params = optimize_for_target(models, target_impulse, args.max_pressure, seed=args.seed)

        try:
            match = compare_curves(params, target_time, target_thrust=target_thrust)
            cosine_sim = match.get('thrust_cosine_similarity')
            rel_err = match.get('thrust_relative_impulse_error_pct')
            real_impulse = match['candidate_sim_result']['total_impulse']
            print(f"cosine_sim={cosine_sim:.4f}, impulse_err={rel_err:.2f}%, "
                  f"real_impulse={real_impulse:.0f}Ns")
            results.append({
                'target_impulse': target_impulse, 'params': params,
                'cosine_similarity': cosine_sim, 'impulse_error_pct': rel_err,
                'real_impulse': real_impulse,
            })
        except Exception as e:
            print(f"FAILED - {e}")

    if not results:
        print("\nNo successful runs - check that motorlib is importable (PYTHONPATH set).")
        return

    results.sort(key=lambda r: -r['cosine_similarity'])
    best = results[0]

    print(f"\n=== Best match found ===")
    print(f"Target impulse: {best['target_impulse']:.1f} Ns "
          f"(real: {best['real_impulse']:.0f} Ns)")
    print(f"Cosine similarity: {best['cosine_similarity']:.4f}")
    print(f"Impulse error: {best['impulse_error_pct']:.2f}%")
    print(f"Params: {best['params']}")

    print(f"\n=== Full ranking ===")
    for r in results:
        print(f"  target={r['target_impulse']:.0f}Ns  "
              f"cosine_sim={r['cosine_similarity']:.4f}  "
              f"impulse_err={r['impulse_error_pct']:.2f}%")


if __name__ == '__main__':
    main()