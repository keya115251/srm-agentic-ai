"""
Design Agent v2: replaces the v1 random-search loop with scipy's
differential_evolution optimizer. Same surrogate model, same scoring idea
(closeness to target impulse + penalties for violating pressure/thrust
constraints) but now the search actively improves candidates round over
round instead of relying on brute-force random sampling.

Usage:
    python design_agent/search.py --target-impulse 1300 --max-pressure 6e6

Requires:
    - surrogate/surrogate_model.joblib (from surrogate/train_surrogate.py)
    - motorlib importable for the final validation step (optional - skipped
      with a warning if not available, same as v1)
"""
import argparse
import os
import sys

import joblib
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SURROGATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'surrogate', 'surrogate_model.joblib'
)

# Same trusted envelope as the training sweep and v1 search - the surrogate's
# predictions are only reliable inside these bounds.
DIAMETER_BOUNDS = (0.06, 0.10)
LENGTH_BOUNDS = (0.10, 0.18)
CORE_DIAMETER_FRAC_BOUNDS = (0.30, 0.75)  # fraction of diameter
THROAT_BOUNDS = (0.010, 0.018)
EXIT_THROAT_RATIO_BOUNDS = (3.0, 9.0)      # multiple of throat

# DE optimizes over these 5 "raw" variables directly. The last two
# (core fraction, exit ratio) get converted to actual coreDiameter/exit
# values inside decode_params() so every candidate DE produces is
# geometrically valid by construction, same guarantee v1 had.
BOUNDS = [
    DIAMETER_BOUNDS,
    LENGTH_BOUNDS,
    CORE_DIAMETER_FRAC_BOUNDS,
    THROAT_BOUNDS,
    EXIT_THROAT_RATIO_BOUNDS,
]


def decode_params(x: np.ndarray) -> dict:
    diameter, length, core_frac, throat, exit_ratio = x
    return {
        'diameter': diameter,
        'length': length,
        'coreDiameter': diameter * core_frac,
        'throat': throat,
        'exit': throat * exit_ratio,
    }


def make_objective(models: dict, target_impulse: float,
                    max_pressure: float = None, max_thrust: float = None):
    """Returns a function DE will minimize: lower = better design."""
    feature_order = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']

    def objective(x: np.ndarray) -> float:
        params = decode_params(x)
        X = np.array([[params[f] for f in feature_order]])

        impulse = models['total_impulse'].predict(X)[0]
        pressure = models['peak_pressure'].predict(X)[0]
        thrust = models['peak_thrust'].predict(X)[0]

        score = abs(impulse - target_impulse) / target_impulse

        if max_pressure is not None and pressure > max_pressure:
            score += 5.0 * (pressure - max_pressure) / max_pressure
        if max_thrust is not None and thrust > max_thrust:
            score += 5.0 * (thrust - max_thrust) / max_thrust

        return score

    return objective


def explain_result(params: dict, models: dict, target_impulse: float) -> str:
    feature_order = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']
    X = np.array([[params[f] for f in feature_order]])
    impulse = models['total_impulse'].predict(X)[0]
    thrust = models['peak_thrust'].predict(X)[0]
    pressure = models['peak_pressure'].predict(X)[0]
    kn = models['peak_kn'].predict(X)[0]
    core_frac = params['coreDiameter'] / params['diameter']

    return (
        f"diameter={params['diameter']*1000:.1f}mm, "
        f"length={params['length']*1000:.1f}mm, "
        f"coreDiameter={params['coreDiameter']*1000:.1f}mm ({core_frac:.0%} of diameter), "
        f"throat={params['throat']*1000:.2f}mm, exit={params['exit']*1000:.1f}mm\n"
        f"    Predicted: impulse={impulse:.0f}Ns (target {target_impulse:.0f}Ns, "
        f"{abs(impulse-target_impulse)/target_impulse:.1%} off), "
        f"peak_thrust={thrust:.0f}N, peak_pressure={pressure/1e6:.2f}MPa, peak_kn={kn:.0f}"
    )


def validate_with_real_sim(params: dict):
    try:
        from simulator_bridge.bates_motor import run_bates_sim
    except ImportError:
        print("\n[!] motorlib not importable - skipping real-simulator "
              "validation. Set PYTHONPATH to your openMotor clone to "
              "enable this step. Surrogate prediction above is UNVERIFIED.")
        return None
    try:
        return run_bates_sim(params)
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-impulse', type=float, required=True)
    parser.add_argument('--max-pressure', type=float, default=None)
    parser.add_argument('--max-thrust', type=float, default=None)
    parser.add_argument('--popsize', type=int, default=20,
                         help='DE population size multiplier (scipy default scaling)')
    parser.add_argument('--maxiter', type=int, default=100,
                         help='max DE generations')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SURROGATE_PATH):
        print(f"ERROR: surrogate model not found at {SURROGATE_PATH}. "
              f"Run surrogate/train_surrogate.py first.")
        return

    d = joblib.load(SURROGATE_PATH)
    models = d['models']

    objective = make_objective(models, args.target_impulse,
                                args.max_pressure, args.max_thrust)

    result = differential_evolution(
        objective, BOUNDS,
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed, tol=1e-6, polish=True,
    )

    best_params = decode_params(result.x)

    print(f"DE converged after {result.nit} generations, "
          f"{result.nfev} surrogate evaluations (score={result.fun:.4f}).")
    print(f"Target: impulse={args.target_impulse:.0f}Ns"
          + (f", max_pressure={args.max_pressure/1e6:.2f}MPa" if args.max_pressure else "")
          + (f", max_thrust={args.max_thrust:.0f}N" if args.max_thrust else ""))
    print(f"\nBest design found:")
    print(f"  {explain_result(best_params, models, args.target_impulse)}")

    validated = validate_with_real_sim(best_params)
    if validated is not None:
        if 'error' in validated:
            print(f"\nReal-simulator validation FAILED: {validated['error']}")
        else:
            surrogate_impulse = models['total_impulse'].predict(
                np.array([[best_params[f] for f in
                           ['diameter','length','coreDiameter','throat','exit']]])
            )[0]
            real_impulse = validated['total_impulse']
            drift = abs(real_impulse - surrogate_impulse) / real_impulse
            print(f"\n=== Real-simulator validation ===")
            print(f"Real impulse={real_impulse:.0f}Ns "
                  f"(surrogate said {surrogate_impulse:.0f}Ns, {drift:.1%} drift), "
                  f"real peak_pressure={validated['peak_pressure']/1e6:.2f}MPa, "
                  f"alerts={validated['alerts']}")


if __name__ == '__main__':
    main()