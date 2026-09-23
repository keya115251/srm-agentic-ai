"""
Design Agent v3 (hybrid): global search via differential_evolution followed
by local refinement via Nelder-Mead. This mirrors the "HO" (hybrid
optimization) approach from Nisar et al.'s FINOCYL grain paper, where GA
finds a promising neighborhood globally and a local optimizer polishes it -
but with one important correction to their approach, explained below.

IMPORTANT - why Nelder-Mead and not SLSQP/gradient-based methods:
This surrogate is a GradientBoostingRegressor, which produces a piecewise-
constant (step-function) prediction surface, not a smooth one. Gradient-
based local optimizers (SLSQP, L-BFGS, the SQP used in Nisar et al.'s paper)
estimate the local slope via finite differences, and a tiny finite-difference
step very often lands inside the same flat "step," measuring an apparent
gradient of exactly zero even when better designs exist a bit further away.
We verified this directly: SLSQP was reporting false convergence after a
single iteration regardless of how far DE's answer was from optimal.

Nelder-Mead (and Powell) don't need gradients at all - they compare actual
function values at nearby points, so they see the true staircase shape and
can still walk toward better designs. This is the correct pairing for a
tree-based surrogate; a smooth surrogate (e.g. a neural network) would not
have this issue and could use SLSQP safely instead.

Usage:
    python design_agent/search_v3_hybrid.py --target-impulse 1300 --max-pressure 6e6
"""
import argparse
import os
import sys

import joblib
import numpy as np
from scipy.optimize import differential_evolution, minimize

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

BOUNDS = [
    DIAMETER_BOUNDS,
    LENGTH_BOUNDS,
    CORE_DIAMETER_FRAC_BOUNDS,
    THROAT_BOUNDS,
    EXIT_THROAT_RATIO_BOUNDS,
]

FEATURE_ORDER = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']


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
    def objective(x: np.ndarray) -> float:
        params = decode_params(x)
        X = np.array([[params[f] for f in FEATURE_ORDER]])

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
    X = np.array([[params[f] for f in FEATURE_ORDER]])
    impulse = models['total_impulse'].predict(X)[0]
    thrust = models['peak_thrust'].predict(X)[0]
    pressure = models['peak_pressure'].predict(X)[0]
    kn = models['peak_kn'].predict(X)[0]
    core_frac = params['coreDiameter'] / params['diameter']

    return (
        f"diameter={params['diameter']*1000:.2f}mm, "
        f"length={params['length']*1000:.2f}mm, "
        f"coreDiameter={params['coreDiameter']*1000:.2f}mm ({core_frac:.1%} of diameter), "
        f"throat={params['throat']*1000:.3f}mm, exit={params['exit']*1000:.2f}mm\n"
        f"    Predicted: impulse={impulse:.1f}Ns (target {target_impulse:.0f}Ns, "
        f"{abs(impulse-target_impulse)/target_impulse:.2%} off), "
        f"peak_thrust={thrust:.1f}N, peak_pressure={pressure/1e6:.3f}MPa, peak_kn={kn:.1f}"
    )


def validate_with_real_sim(params: dict):
    try:
        from simulator_bridge.bates_motor import run_bates_sim
    except ImportError:
        print("\n[!] motorlib not importable - skipping real-simulator "
              "validation. Set PYTHONPATH to your openMotor clone.")
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
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SURROGATE_PATH):
        print(f"ERROR: surrogate model not found at {SURROGATE_PATH}.")
        return

    d = joblib.load(SURROGATE_PATH)
    models = d['models']
    objective = make_objective(models, args.target_impulse,
                                args.max_pressure, args.max_thrust)

    # --- Stage 1: DE global search ---
    de_result = differential_evolution(
        objective, BOUNDS,
        popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed, tol=1e-6, polish=False,  # polish=False: we do our own polishing below
    )
    de_params = decode_params(de_result.x)
    de_score = de_result.fun

    print(f"=== Stage 1: Differential Evolution (global search) ===")
    print(f"Converged after {de_result.nit} generations, "
          f"{de_result.nfev} surrogate evaluations, score={de_score:.6f}")
    print(f"  {explain_result(de_params, models, args.target_impulse)}")

    # --- Stage 2: Nelder-Mead local refinement, starting from DE's answer ---
    # NOT SLSQP - see module docstring for why gradient-based methods fail
    # on this tree-based surrogate's piecewise-constant prediction surface.
    nm_result = minimize(
        objective, x0=de_result.x, method='Nelder-Mead',
        bounds=BOUNDS, options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
    )
    nm_params = decode_params(nm_result.x)
    nm_score = nm_result.fun

    print(f"\n=== Stage 2: Nelder-Mead local refinement (starting from DE's answer) ===")
    print(f"Converged: {nm_result.success}, {nm_result.nit} iterations, "
          f"{nm_result.nfev} additional surrogate evaluations, score={nm_score:.6f}")
    print(f"  {explain_result(nm_params, models, args.target_impulse)}")

    improvement = de_score - nm_score
    print(f"\nNelder-Mead improved the score by {improvement:.6f} "
          f"({'meaningful refinement' if improvement > 1e-5 else 'negligible - DE was already near-optimal'})")

    total_evals = de_result.nfev + nm_result.nfev
    print(f"Total surrogate evaluations across both stages: {total_evals}")

    # Use whichever stage actually produced the better (lower) score
    final_params = nm_params if nm_score <= de_score else de_params
    final_score = min(nm_score, de_score)

    print(f"\n=== Final design (best of both stages, score={final_score:.6f}) ===")
    print(f"  {explain_result(final_params, models, args.target_impulse)}")

    validated = validate_with_real_sim(final_params)
    if validated is not None:
        if 'error' in validated:
            print(f"\nReal-simulator validation FAILED: {validated['error']}")
        else:
            X = np.array([[final_params[f] for f in FEATURE_ORDER]])
            surrogate_impulse = models['total_impulse'].predict(X)[0]
            real_impulse = validated['total_impulse']
            drift = abs(real_impulse - surrogate_impulse) / real_impulse
            print(f"\n=== Real-simulator validation ===")
            print(f"Real impulse={real_impulse:.1f}Ns "
                  f"(surrogate said {surrogate_impulse:.1f}Ns, {drift:.2%} drift), "
                  f"real peak_pressure={validated['peak_pressure']/1e6:.3f}MPa, "
                  f"alerts={validated['alerts']}")


if __name__ == '__main__':
    main()