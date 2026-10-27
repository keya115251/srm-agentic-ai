"""
K-class sweep across all four grain types: the kclass_sweep.py idea (sweep
impulse targets across 1280-2560 Ns and check each resulting design's thrust
curve against a reference curve such as the real K950), with the design for
each target found by search_multigrain.py's per-grain search, so BATES,
Finocyl, the BATES + Finocyl hybrid and Star are compared on curve shape.

For every (grain, target) pair the chosen design is simulated for real and
its thrust curve compared with the reference by curve_matching.thrust_match(),
which scores absolute thrust (normalized RMSE plus impulse and burn time
error) rather than shape alone; cosine similarity is still printed for
reference. A design is eligible for "best match" only if its REAL
impulse is inside K-class and its real simulation raises no alert other than
the tail-off flow-separation warning (see search_multigrain.py) and it
passes the structural checks at its real peak pressure
(structural_tools/casing.py; same --case-wall-mm etc. options).

Usage:
    python design_agent/kclass_sweep_multigrain.py --target-csv data/k950_target_curve.csv
    python design_agent/kclass_sweep_multigrain.py --target-csv data/k950_target_curve.csv \
        --grains Star Finocyl --n-points 5
"""
import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import design_agent.search_multigrain as sm
from design_agent.curve_matching import describe_match, thrust_match
from design_agent.kclass_sweep import K_CLASS_MIN, K_CLASS_MAX
from structural_tools.casing import add_structural_args, case_spec_from_args

warnings.filterwarnings('ignore', category=RuntimeWarning, module='motorlib')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-csv', required=True,
                        help='CSV with the reference curve (columns: time, thrust)')
    parser.add_argument('--max-pressure', type=float, default=6e6)
    parser.add_argument('--n-points', type=int, default=9,
                        help='number of impulse targets across the K-class range')
    parser.add_argument('--grains', nargs='+', default=list(sm.GRAINS), choices=list(sm.GRAINS))
    parser.add_argument('--popsize', type=int, default=15)
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--save-json', default=None,
                        help='write every (grain, target) result, with curves, here')
    add_structural_args(parser)
    args = parser.parse_args()
    case_spec = case_spec_from_args(args)

    with open(args.target_csv) as f:
        rows = list(csv.DictReader(f))
    target_time = np.array([float(r['time']) for r in rows])
    target_thrust = np.array([float(r['thrust']) for r in rows])
    impulse_targets = np.linspace(K_CLASS_MIN, K_CLASS_MAX, args.n_points)

    sm._sim_counter = multiprocessing.Value('i', 0)
    pool = (multiprocessing.Pool(args.workers, initializer=sm._init_worker,
                                 initargs=(sm._sim_counter,))
            if args.workers > 1 else None)

    print(f"Sweeping {args.n_points} impulse targets ({K_CLASS_MIN:.0f}-{K_CLASS_MAX:.0f} Ns) "
          f"x {len(args.grains)} grain types, max pressure {args.max_pressure / 1e6:.1f}MPa\n")
    results = []
    try:
        for name in args.grains:
            for i, target in enumerate(impulse_targets, 1):
                print(f"[{name} {i}/{len(impulse_targets)}] target={target:.0f}Ns ... ",
                      end='', flush=True)
                t0 = time.time()
                search_args = SimpleNamespace(
                    target_impulse=target, max_pressure=args.max_pressure,
                    max_thrust=None, popsize=args.popsize, maxiter=args.maxiter,
                    seed=args.seed, case_spec=case_spec)
                try:
                    r = sm.search_grain(name, search_args, pool)
                    sim = sm.GRAINS[name].simulate(r['params'])
                except Exception as e:
                    print(f"FAILED - {e}")
                    continue
                real = sim['total_impulse']
                match = thrust_match(sim['time'], sim['thrust'], target_time, target_thrust)
                row = {
                    'grain': name, 'target_impulse': float(target), 'params': r['params'],
                    'pred_impulse': r['pred']['total_impulse'], 'real_impulse': real,
                    'drift': abs(real - r['pred']['total_impulse']) / real,
                    'peak_pressure': sim['peak_pressure'],
                    'match_error': match['match_error'], 'match': match,
                    'cosine_similarity': match['cosine_similarity'],
                    'valid': K_CLASS_MIN <= real <= K_CLASS_MAX,
                    # search_grain() re-simulates the same params, so its
                    # structural verdict holds for `sim` too.
                    'flagged': r['flagged'], 'alerts': sim['alerts'],
                    'structural_failed': (r['structural'] or {}).get('failed', []),
                    'fallback_sims': r['fallback_sims'],
                    'curve': {k: sim[k] for k in ('time', 'thrust', 'pressure', 'kn')},
                }
                results.append(row)
                status = ('FLAGGED' if row['flagged'] else
                          'OK' if row['valid'] else 'INVALID (outside K-class)')
                print(f"match_error={row['match_error']:.4f}  real={real:.0f}Ns  "
                      f"drift={row['drift']:.1%}  {r['fallback_sims']} fallback sims, "
                      f"{time.time() - t0:.0f}s  {status}", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    if not results:
        print("\nNo successful runs - check that motorlib is importable (PYTHONPATH set).")
        return

    eligible = [r for r in results if r['valid'] and not r['flagged']]
    print("\n=== Best eligible match per grain type (lowest thrust match error) ===")
    for name in args.grains:
        mine = [r for r in eligible if r['grain'] == name]
        n_all = sum(r['grain'] == name for r in results)
        if not mine:
            print(f"  {name:<8} no eligible design ({n_all} run)")
            continue
        b = min(mine, key=lambda r: r['match_error'])
        print(f"  {name:<8} match_error={b['match_error']:.4f}  "
              f"cos_sim={b['cosine_similarity']:.4f}  target={b['target_impulse']:.0f}Ns  "
              f"real={b['real_impulse']:.0f}Ns  peak_pressure={b['peak_pressure'] / 1e6:.2f}MPa  "
              f"({len(mine)}/{n_all} eligible)")

    if eligible:
        best = min(eligible, key=lambda r: r['match_error'])
        print(f"\n=== Best overall: {best['grain']} ===")
        print(f"Vs reference: {describe_match(best['match'])}")
        print(f"Target {best['target_impulse']:.0f}Ns, real {best['real_impulse']:.0f}Ns "
              f"(drift {best['drift']:.1%}), alerts: {best['alerts'] or 'none'}")
        print(f"Params: {sm.fmt_params(best['params'])}")
    else:
        print("\nNo eligible K-class design for any grain type.")

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, default=float)
        print(f"Saved results to {args.save_json}")


if __name__ == '__main__':
    main()
