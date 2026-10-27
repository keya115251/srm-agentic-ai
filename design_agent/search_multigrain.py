"""
Design Agent v4 (multi-grain): runs the v3 hybrid search (differential
evolution, then Nelder-Mead) once per grain type - BATES, Finocyl, the
BATES + Finocyl hybrid, and Star - and compares the best design each grain
type can reach for the same targets.

Every candidate is scored through that grain's range-guarded surrogate,
predict(params, fallback=True): inside the surrogate's training ranges the
prediction takes milliseconds; outside them the real simulator runs instead,
so the optimizer can never exploit a flat, extrapolated surrogate region.
Those fallback simulations are counted and reported, since each one costs a
few seconds.

Each grain's final pick is then re-simulated for real, and the grains are
ranked on the real results, not the surrogate's. A design whose real
simulation raises an alert other than the tail-off flow-separation warning
(which every Finocyl, hybrid and Star design trips during sliver burnout) is
marked FLAGGED and not eligible as the overall pick. During the search, a
candidate that falls back to the simulator and raises such an alert is
penalized, since otherwise the exact fallback result draws the optimizer
into flagged regions (the first 1300 Ns run picked a hybrid with port/throat
1.3 and a mass-flux alert this way). Note the surrogates
differ here: the BATES surrogate was trained on all successful sims,
alert-flagged ones included, while the Finocyl, hybrid and Star surrogates
were trained on clean sims only, so their predictions for designs that would
trip an alert are less reliable. The real re-simulation is what catches that.

Every candidate is also checked structurally (structural_tools/casing.py:
case wall, burst, forward bulkhead and closure bolts at the predicted peak
pressure); a negative margin of safety is penalized like a pressure
overshoot, and a final pick that fails the check at its real peak pressure
is FLAGGED. The case hardware comes from --case-wall-mm, --bulkhead-mm,
--n-bolts, --bolt-mm and --safety-factor; --no-structural turns it off.

The search space for each grain is the space its training data was sampled
from (the bounds constants in scripts/generate_sweep_data_*.py, imported
below so they can't drift), parameterized the same way: dependent dimensions
such as core diameter or fin length are searched as fractions of the room
available to them.

Usage:
    python design_agent/search_multigrain.py --target-impulse 1300 --max-pressure 6e6
    python design_agent/search_multigrain.py --target-impulse 1300 --grains Star Finocyl
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.generate_sweep_data_random as bates_gen
import scripts.generate_sweep_data_finocyl as finocyl_gen
import scripts.generate_sweep_data_hybrid as hybrid_gen
import scripts.generate_sweep_data_star as star_gen
from simulator_bridge.bates_motor import run_bates_sim
from simulator_bridge.finocyl_motor import run_finocyl_sim
from simulator_bridge.hybrid_motor import run_hybrid_sim
from simulator_bridge.star_motor import run_star_sim
from structural_tools.casing import (DEFAULT_CASE, add_structural_args, case_spec_from_args,
                                     check_structure, describe)
from surrogate import bates_surrogate, finocyl_surrogate, hybrid_surrogate, star_surrogate

warnings.filterwarnings('ignore', category=RuntimeWarning, module='motorlib')

SEPARATION_ALERT = 'Low exit pressure, nozzle flow may separate'
# Added to the score of any candidate whose fallback simulation raises an
# alert other than SEPARATION_ALERT. Larger than any impulse error the
# search can reach inside the bounds, so a clean design always wins.
FLAG_PENALTY = 1.0


def _int_bounds(b):
    # Integer dimensions are searched over [lo, hi + 1) and floored, so each
    # integer gets an equal share of the interval.
    return (b[0], b[1] + 1 - 1e-9)


# --- Decoders: optimizer vector -> wrapper params, mirroring each
# generator's sample_params() ---

def decode_bates(x):
    diameter, length, core_frac, throat, exit_ratio = x
    return {'diameter': diameter, 'length': length,
            'coreDiameter': diameter * core_frac,
            'throat': throat, 'exit': throat * exit_ratio}


def decode_finocyl(x):
    diameter, length, core_frac, num_fins, fin_l_frac, fin_w_frac, throat, exit_ratio = x
    num_fins = int(math.floor(num_fins))
    core = diameter * core_frac
    return {'diameter': diameter, 'length': length, 'coreDiameter': core,
            'numFins': num_fins,
            'finWidth': math.pi * core / num_fins * fin_w_frac,
            'finLength': (diameter - core) / 2 * fin_l_frac,
            'throat': throat, 'exit': throat * exit_ratio}


def decode_hybrid(x):
    (diameter, fin_seg_len, fin_core_frac, num_fins, fin_l_frac, fin_w_frac,
     bates_len, bates_core_frac, throat, exit_ratio) = x
    num_fins = int(math.floor(num_fins))
    fin_core = diameter * fin_core_frac
    return {'diameter': diameter, 'finocylLength': fin_seg_len,
            'finocylCoreDiameter': fin_core, 'numFins': num_fins,
            'finWidth': math.pi * fin_core / num_fins * fin_w_frac,
            'finLength': (diameter - fin_core) / 2 * fin_l_frac,
            'batesLength': bates_len,
            'batesCoreDiameter': diameter * bates_core_frac,
            'throat': throat, 'exit': throat * exit_ratio}


def decode_star(x):
    diameter, length, num_points, pl_frac, pw_frac, throat, exit_ratio = x
    point_length = diameter / 2 * pl_frac
    return {'diameter': diameter, 'length': length,
            'numPoints': int(math.floor(num_points)),
            'pointLength': point_length,
            'pointWidth': point_length * pw_frac,
            'throat': throat, 'exit': throat * exit_ratio}


@dataclass
class Grain:
    predict: Callable
    simulate: Callable
    decode: Callable
    bounds: List[Tuple[float, float]]
    integer_dims: Tuple[int, ...] = ()


GRAINS = {
    'BATES': Grain(
        bates_surrogate.predict, run_bates_sim, decode_bates,
        [bates_gen.DIAMETER_BOUNDS, bates_gen.LENGTH_BOUNDS,
         bates_gen.CORE_DIAMETER_FRAC_BOUNDS, bates_gen.THROAT_BOUNDS,
         bates_gen.EXIT_THROAT_RATIO_BOUNDS]),
    'Finocyl': Grain(
        finocyl_surrogate.predict, run_finocyl_sim, decode_finocyl,
        [finocyl_gen.DIAMETER_BOUNDS, finocyl_gen.LENGTH_BOUNDS,
         finocyl_gen.CORE_DIAMETER_FRAC_BOUNDS, _int_bounds(finocyl_gen.NUM_FINS_BOUNDS),
         finocyl_gen.FIN_LENGTH_FRAC_BOUNDS, finocyl_gen.FIN_WIDTH_FRAC_BOUNDS,
         finocyl_gen.THROAT_BOUNDS, finocyl_gen.EXIT_THROAT_RATIO_BOUNDS],
        integer_dims=(3,)),
    'Hybrid': Grain(
        hybrid_surrogate.predict, run_hybrid_sim, decode_hybrid,
        [hybrid_gen.DIAMETER_BOUNDS, hybrid_gen.FINOCYL_LENGTH_BOUNDS,
         hybrid_gen.FINOCYL_CORE_FRAC_BOUNDS, _int_bounds(hybrid_gen.NUM_FINS_BOUNDS),
         hybrid_gen.FIN_LENGTH_FRAC_BOUNDS, hybrid_gen.FIN_WIDTH_FRAC_BOUNDS,
         hybrid_gen.BATES_LENGTH_BOUNDS, hybrid_gen.BATES_CORE_FRAC_BOUNDS,
         hybrid_gen.THROAT_BOUNDS, hybrid_gen.EXIT_THROAT_RATIO_BOUNDS],
        integer_dims=(3,)),
    'Star': Grain(
        star_surrogate.predict, run_star_sim, decode_star,
        [star_gen.DIAMETER_BOUNDS, star_gen.LENGTH_BOUNDS,
         _int_bounds(star_gen.NUM_POINTS_BOUNDS),
         star_gen.POINT_LENGTH_FRAC_BOUNDS, star_gen.POINT_WIDTH_FRAC_BOUNDS,
         star_gen.THROAT_BOUNDS, star_gen.EXIT_THROAT_RATIO_BOUNDS],
        integer_dims=(2,)),
}


def score(pred: dict, target_impulse: float, max_pressure: float = None,
          max_thrust: float = None) -> float:
    """Same objective as search_v3_hybrid.py: relative impulse error plus a
    5x penalty on any pressure/thrust overshoot."""
    s = abs(pred['total_impulse'] - target_impulse) / target_impulse
    if max_pressure is not None and pred['peak_pressure'] > max_pressure:
        s += 5.0 * (pred['peak_pressure'] - max_pressure) / max_pressure
    if max_thrust is not None and pred['peak_thrust'] > max_thrust:
        s += 5.0 * (pred['peak_thrust'] - max_thrust) / max_thrust
    return s


# Fallback-simulation counter, shared with DE's worker processes.
_sim_counter = None


def _init_worker(counter):
    global _sim_counter
    _sim_counter = counter
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def objective(x, grain_name, target_impulse, max_pressure, max_thrust, case_spec=None):
    grain = GRAINS[grain_name]
    params = grain.decode(x)
    pred = grain.predict(params, fallback=True)
    s = score(pred, target_impulse, max_pressure, max_thrust)
    if case_spec is not None:
        # Same 5x weighting as a pressure overshoot: a negative margin of
        # safety is the fraction by which the worst stress exceeds its
        # allowable at the predicted peak pressure.
        st = check_structure(params, pred['peak_pressure'], case_spec)
        s += 5.0 * max(0.0, -st['min_margin'])
    if pred['source'] == 'simulator':
        if _sim_counter is not None:
            with _sim_counter.get_lock():
                _sim_counter.value += 1
        # A fallback sim is exact, so without this the optimizer settles on
        # out-of-range designs the real simulator flags (port/throat, mass
        # flux) - which is why they are out of the clean-trained range.
        if is_flagged(pred['alerts']):
            s += FLAG_PENALTY
    return s


def is_flagged(alerts: List[str]) -> bool:
    return any(a != SEPARATION_ALERT for a in alerts)


def fmt_params(params: dict) -> str:
    return ', '.join(f"{k}={v}" if isinstance(v, int) else f"{k}={v * 1000:.2f}mm"
                     for k, v in params.items())


def search_grain(name: str, args, pool) -> dict:
    grain = GRAINS[name]
    case_spec = getattr(args, 'case_spec', DEFAULT_CASE)
    obj = partial(objective, grain_name=name, target_impulse=args.target_impulse,
                  max_pressure=args.max_pressure, max_thrust=args.max_thrust,
                  case_spec=case_spec)
    start_sims = _sim_counter.value
    t0 = time.time()

    de = differential_evolution(
        obj, grain.bounds, popsize=args.popsize, maxiter=args.maxiter,
        seed=args.seed, tol=1e-6, polish=False,
        updating='deferred', workers=pool.map if pool else 1)
    # Nelder-Mead refinement from DE's answer (see search_v3_hybrid.py for
    # why not a gradient method on a tree surrogate). Integer dimensions are
    # floored by the decoder, so NM moving them continuously is harmless.
    nm = minimize(obj, x0=de.x, method='Nelder-Mead', bounds=grain.bounds,
                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    x = nm.x if nm.fun <= de.fun else de.x
    params = grain.decode(x)
    pred = grain.predict(params, fallback=True)

    real = grain.simulate(params)
    real_score = score(real, args.target_impulse, args.max_pressure, args.max_thrust)
    structural = (check_structure(params, real['peak_pressure'], case_spec)
                  if case_spec is not None else None)
    return {
        'grain': name, 'params': params,
        'evals': de.nfev + nm.nfev, 'fallback_sims': _sim_counter.value - start_sims,
        'search_seconds': time.time() - t0,
        'surrogate_score': float(min(nm.fun, de.fun)), 'pred': pred,
        'real': {k: real[k] for k in ('total_impulse', 'peak_thrust',
                                      'peak_pressure', 'peak_kn', 'burn_time')},
        'alerts': real['alerts'], 'structural': structural,
        'flagged': is_flagged(real['alerts']) or bool(structural and not structural['ok']),
        'real_score': real_score,
    }


def print_result(r: dict, target_impulse: float):
    p, real = r['pred'], r['real']
    drift = (real['total_impulse'] - p['total_impulse']) / real['total_impulse']
    print(f"\n=== {r['grain']} ===")
    print(f"  {r['evals']} evaluations, {r['fallback_sims']} of them fell back to the "
          f"real simulator, {r['search_seconds']:.0f}s")
    print(f"  {fmt_params(r['params'])}")
    print(f"  Predicted ({p['source']}): impulse={p['total_impulse']:.1f}Ns, "
          f"peak_thrust={p['peak_thrust']:.1f}N, peak_pressure={p['peak_pressure'] / 1e6:.3f}MPa")
    print(f"  Real sim:  impulse={real['total_impulse']:.1f}Ns "
          f"({abs(real['total_impulse'] - target_impulse) / target_impulse:.2%} off target, "
          f"surrogate drift {drift:+.2%}), peak_thrust={real['peak_thrust']:.1f}N, "
          f"peak_pressure={real['peak_pressure'] / 1e6:.3f}MPa, burn_time={real['burn_time']:.2f}s")
    if r['structural'] is not None:
        print(f"  Case ID {r['structural']['case_inner_diameter'] * 1000:.1f}mm: "
              f"{describe(r['structural'])}")
    print(f"  Alerts: {r['alerts'] or 'none'}{'  -> FLAGGED' if r['flagged'] else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-impulse', type=float, required=True)
    parser.add_argument('--max-pressure', type=float, default=None)
    parser.add_argument('--max-thrust', type=float, default=None)
    parser.add_argument('--grains', nargs='+', default=list(GRAINS), choices=list(GRAINS))
    parser.add_argument('--popsize', type=int, default=15)
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help='processes for DE population evaluation (fallback sims are slow)')
    parser.add_argument('--save-json', default=None, help='write per-grain results here')
    add_structural_args(parser)
    args = parser.parse_args()
    args.case_spec = case_spec_from_args(args)

    global _sim_counter
    _sim_counter = multiprocessing.Value('i', 0)
    pool = (multiprocessing.Pool(args.workers, initializer=_init_worker,
                                 initargs=(_sim_counter,))
            if args.workers > 1 else None)

    print(f"Target impulse {args.target_impulse:.0f}Ns"
          + (f", max pressure {args.max_pressure / 1e6:.2f}MPa" if args.max_pressure else '')
          + (f", max thrust {args.max_thrust:.0f}N" if args.max_thrust else ''))
    results = []
    try:
        for name in args.grains:
            r = search_grain(name, args, pool)
            print_result(r, args.target_impulse)
            results.append(r)
    finally:
        if pool:
            pool.close()
            pool.join()

    print("\n=== Ranking by real-simulator score (lower is better) ===")
    ranked = sorted(results, key=lambda r: (r['flagged'], r['real_score']))
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['grain']:<8} real score={r['real_score']:.4f}  "
              f"impulse={r['real']['total_impulse']:.1f}Ns  "
              f"peak_pressure={r['real']['peak_pressure'] / 1e6:.3f}MPa"
              f"{'  FLAGGED' if r['flagged'] else ''}")
    clean = [r for r in ranked if not r['flagged']]
    if clean:
        print(f"\nBest grain type for this target: {clean[0]['grain']}")
    else:
        print("\nEvery grain type's pick raised an alert in the real simulator.")

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Saved results to {args.save_json}")


if __name__ == '__main__':
    main()
