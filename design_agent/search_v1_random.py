"""
Design Agent v1 (random search) - kept alongside search.py (now v2, using
differential_evolution) purely so you can run both against the same target
and compare directly: number of surrogate evaluations needed, final score,
and real-simulator validation drift.

Usage:
    python design_agent/search_v1_random.py --target-impulse 1300 --max-pressure 6e6
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SURROGATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'surrogate', 'surrogate_model.joblib'
)

DIAMETER_BOUNDS = (0.06, 0.10)
LENGTH_BOUNDS = (0.10, 0.18)
CORE_DIAMETER_FRAC_BOUNDS = (0.30, 0.75)
THROAT_BOUNDS = (0.010, 0.018)
EXIT_THROAT_RATIO_BOUNDS = (3.0, 9.0)


def sample_candidates(n: int, rng: np.random.Generator) -> np.ndarray:
    diameter = rng.uniform(*DIAMETER_BOUNDS, size=n)
    length = rng.uniform(*LENGTH_BOUNDS, size=n)
    core_frac = rng.uniform(*CORE_DIAMETER_FRAC_BOUNDS, size=n)
    core_diameter = diameter * core_frac
    throat = rng.uniform(*THROAT_BOUNDS, size=n)
    exit_ratio = rng.uniform(*EXIT_THROAT_RATIO_BOUNDS, size=n)
    exit_dia = throat * exit_ratio
    return np.column_stack([diameter, length, core_diameter, throat, exit_dia])


def score_candidates(X: np.ndarray, models: dict, target_impulse: float,
                      max_pressure: float = None, max_thrust: float = None) -> dict:
    preds = {name: model.predict(X) for name, model in models.items()}
    impulse_error = np.abs(preds['total_impulse'] - target_impulse) / target_impulse
    score = impulse_error.copy()
    penalty = np.zeros(len(X))
    if max_pressure is not None:
        over = np.maximum(0, preds['peak_pressure'] - max_pressure) / max_pressure
        penalty += over * 5.0
    if max_thrust is not None:
        over = np.maximum(0, preds['peak_thrust'] - max_thrust) / max_thrust
        penalty += over * 5.0
    score += penalty
    preds['score'] = score
    return preds


def explain_candidate(params: np.ndarray, preds: dict, idx: int, target_impulse: float) -> str:
    diameter, length, core_diameter, throat, exit_dia = params[idx]
    impulse = preds['total_impulse'][idx]
    thrust = preds['peak_thrust'][idx]
    pressure = preds['peak_pressure'][idx]
    kn = preds['peak_kn'][idx]
    core_frac = core_diameter / diameter
    return (
        f"diameter={diameter*1000:.1f}mm, length={length*1000:.1f}mm, "
        f"coreDiameter={core_diameter*1000:.1f}mm ({core_frac:.0%} of diameter), "
        f"throat={throat*1000:.2f}mm, exit={exit_dia*1000:.1f}mm\n"
        f"    Predicted: impulse={impulse:.0f}Ns (target {target_impulse:.0f}Ns, "
        f"{abs(impulse-target_impulse)/target_impulse:.1%} off), "
        f"peak_thrust={thrust:.0f}N, peak_pressure={pressure/1e6:.2f}MPa, peak_kn={kn:.0f}"
    )


def validate_with_real_sim(params: np.ndarray):
    try:
        from simulator_bridge.bates_motor import run_bates_sim
    except ImportError:
        print("\n[!] motorlib not importable - skipping real-simulator validation.")
        return None
    diameter, length, core_diameter, throat, exit_dia = params
    try:
        return run_bates_sim({
            'diameter': diameter, 'length': length, 'coreDiameter': core_diameter,
            'throat': throat, 'exit': exit_dia,
        })
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-impulse', type=float, required=True)
    parser.add_argument('--max-pressure', type=float, default=None)
    parser.add_argument('--max-thrust', type=float, default=None)
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SURROGATE_PATH):
        print(f"ERROR: surrogate model not found at {SURROGATE_PATH}.")
        return

    d = joblib.load(SURROGATE_PATH)
    models = d['models']

    rng = np.random.default_rng(args.seed)
    X = sample_candidates(args.n_samples, rng)
    preds = score_candidates(X, models, args.target_impulse, args.max_pressure, args.max_thrust)

    best_idx = int(np.argmin(preds['score']))

    print(f"Random search: tried {args.n_samples} candidates "
          f"(score={preds['score'][best_idx]:.4f}).")
    print(f"Target: impulse={args.target_impulse:.0f}Ns"
          + (f", max_pressure={args.max_pressure/1e6:.2f}MPa" if args.max_pressure else "")
          + (f", max_thrust={args.max_thrust:.0f}N" if args.max_thrust else ""))
    print(f"\nBest design found:")
    print(f"  {explain_candidate(X, preds, best_idx, args.target_impulse)}")

    validated = validate_with_real_sim(X[best_idx])
    if validated is not None:
        if 'error' in validated:
            print(f"\nReal-simulator validation FAILED: {validated['error']}")
        else:
            surrogate_impulse = preds['total_impulse'][best_idx]
            real_impulse = validated['total_impulse']
            drift = abs(real_impulse - surrogate_impulse) / real_impulse
            print(f"\n=== Real-simulator validation ===")
            print(f"Real impulse={real_impulse:.0f}Ns "
                  f"(surrogate said {surrogate_impulse:.0f}Ns, {drift:.1%} drift), "
                  f"real peak_pressure={validated['peak_pressure']/1e6:.2f}MPa, "
                  f"alerts={validated['alerts']}")


if __name__ == '__main__':
    main()