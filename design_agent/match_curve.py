"""
Finds the grain design whose thrust curve best matches any target thrust
curve, for all four grain types, scored by curve_matching.thrust_match()
(absolute thrust: normalized RMSE plus impulse and burn time error).

Nothing here depends on the target: impulse, burn time and thrust level all
come from the curve itself, so the same run works for a K950, a G-class
datasheet curve or a test-fire log.

The surrogates only predict scalars, so the search works on real simulated
curves in two stages:

  1. Library screen. Every curve saved from the Latin hypercube runs
     (data/raw/<grain>_sweep_lhs_curves.jsonl and the _holdout file, ~5,500
     per grain) is scored against the target. This takes seconds and needs
     no surrogate. Designs whose simulation raised an alert other than the
     tail-off flow-separation warning, that fail the structural checks
     (structural_tools/casing.py, same --case-wall-mm etc. options) or that
     exceed --max-pressure are not eligible.
  2. Refinement. The best --refine eligible library designs per grain are
     each refined with Nelder-Mead on the real simulator over the grain's
     continuous dimensions (fin and point counts stay fixed), minimizing the
     same match error plus search_multigrain.py's penalties for alerts,
     pressure overshoot and negative structural margin. Each evaluation is a
     full simulation, so this is capped at --refine-evals per start.
     --refine 0 skips it.

Usage:
    python design_agent/match_curve.py --target-csv data/k950_target_curve.csv
    python design_agent/match_curve.py --target-csv my_curve.csv --grains Star BATES \\
        --max-pressure 6e6 --refine 2 --refine-evals 60 --plot outputs/match.png
"""
import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from functools import partial

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import design_agent.search_multigrain as sm
from design_agent.curve_matching import describe_match, thrust_match
from structural_tools.casing import add_structural_args, case_spec_from_args, check_structure

warnings.filterwarnings('ignore', category=RuntimeWarning, module='motorlib')

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'raw')
# Added to the refinement objective when a simulation raises, so the
# optimizer backs away from geometries motorlib cannot run.
SIM_FAIL_PENALTY = 10.0


# --- Encoders: wrapper params -> optimizer vector, the inverses of
# search_multigrain's decoders. Integer dimensions are encoded at the middle
# of their [n, n + 1) interval. ---

def encode_bates(p):
    d = p['diameter']
    return [d, p['length'], p['coreDiameter'] / d, p['throat'], p['exit'] / p['throat']]


def encode_finocyl(p):
    d, core, n = p['diameter'], p['coreDiameter'], p['numFins']
    return [d, p['length'], core / d, n + 0.5,
            p['finLength'] / ((d - core) / 2), p['finWidth'] / (math.pi * core / n),
            p['throat'], p['exit'] / p['throat']]


def encode_hybrid(p):
    d, core, n = p['diameter'], p['finocylCoreDiameter'], p['numFins']
    return [d, p['finocylLength'], core / d, n + 0.5,
            p['finLength'] / ((d - core) / 2), p['finWidth'] / (math.pi * core / n),
            p['batesLength'], p['batesCoreDiameter'] / d,
            p['throat'], p['exit'] / p['throat']]


def encode_star(p):
    d, pl = p['diameter'], p['pointLength']
    return [d, p['length'], p['numPoints'] + 0.5, pl / (d / 2), p['pointWidth'] / pl,
            p['throat'], p['exit'] / p['throat']]


ENCODERS = {'BATES': encode_bates, 'Finocyl': encode_finocyl,
            'Hybrid': encode_hybrid, 'Star': encode_star}


def read_target(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r['time']) for r in rows]),
            np.array([float(r['thrust']) for r in rows]))


def load_library(grain_name: str) -> list:
    """Every successful LHS simulation for a grain (training and holdout
    files), as dicts with params, scalar outputs, alerts and the full curve."""
    grain = sm.GRAINS[grain_name]
    param_names = list(grain.decode(np.mean(grain.bounds, axis=1)))
    entries = []
    for suffix in ('', '_holdout'):
        stem = os.path.join(RAW_DIR, f'{grain_name.lower()}_sweep_lhs{suffix}')
        if not os.path.exists(stem + '.csv'):
            continue
        with open(stem + '.csv') as f:
            rows = {r['sample_id']: r for r in csv.DictReader(f) if r['success'] == 'True'}
        with open(stem + '_curves.jsonl') as f:
            for line in f:
                c = json.loads(line)
                r = rows.get(str(c['sample_id']))
                if r is None:
                    continue
                params = {k: int(r[k]) if k in ('numFins', 'numPoints') else float(r[k])
                          for k in param_names}
                entries.append({
                    'source': f"lhs{suffix or ''}#{c['sample_id']}", 'params': params,
                    'peak_pressure': float(r['peak_pressure']),
                    'alerts': [a for a in r['alerts'].split('; ') if a],
                    'time': c['time'], 'thrust': c['thrust']})
    return entries


def constraint_penalty(params, alerts, peak_pressure, max_pressure, case_spec):
    """search_multigrain.py's penalties, so a refined design can't trade a
    better curve for an alert, a pressure overshoot or a failed case."""
    s = sm.FLAG_PENALTY if sm.is_flagged(alerts) else 0.0
    if max_pressure is not None and peak_pressure > max_pressure:
        s += 5.0 * (peak_pressure - max_pressure) / max_pressure
    if case_spec is not None:
        s += 5.0 * max(0.0, -check_structure(params, peak_pressure, case_spec)['min_margin'])
    return s


def screen(grain_name, target_time, target_thrust, max_pressure, case_spec):
    scored = []
    for e in load_library(grain_name):
        m = thrust_match(e['time'], e['thrust'], target_time, target_thrust)
        e['match'] = m
        e['penalty'] = constraint_penalty(e['params'], e['alerts'], e['peak_pressure'],
                                          max_pressure, case_spec)
        scored.append(e)
    return sorted(scored, key=lambda e: (e['penalty'] > 0, e['match']['match_error']))


def refine(start: dict, grain_name, target_time, target_thrust, max_pressure, case_spec,
           max_evals):
    """Nelder-Mead on the real simulator from one library design, over the
    grain's continuous dimensions."""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    grain = sm.GRAINS[grain_name]
    x0 = np.array(ENCODERS[grain_name](start['params']))
    free = [i for i in range(len(x0)) if i not in grain.integer_dims]
    lo, hi = np.array(grain.bounds).T
    best = {'objective': math.inf}

    def full(z):
        x = x0.copy()
        x[free] = np.clip(z, lo[free], hi[free])
        return x

    def objective(z):
        params = grain.decode(full(z))
        try:
            sim = grain.simulate(params)
            m = thrust_match(sim['time'], sim['thrust'], target_time, target_thrust)
        except Exception:
            return SIM_FAIL_PENALTY
        pen = constraint_penalty(params, sim['alerts'], sim['peak_pressure'],
                                 max_pressure, case_spec)
        obj = m['match_error'] + pen
        if obj < best['objective']:
            best.update(objective=obj, params=params, match=m, penalty=pen,
                        alerts=sim['alerts'], peak_pressure=sim['peak_pressure'],
                        time=sim['time'], thrust=sim['thrust'])
        return obj

    t0 = time.time()
    res = minimize(objective, x0[free], method='Nelder-Mead',
                   bounds=list(zip(lo[free], hi[free])),
                   options={'maxfev': max_evals, 'xatol': 1e-6, 'fatol': 1e-5})
    if 'params' not in best:
        return None
    best.update(source=f"refined from {start['source']}", evals=res.nfev,
                seconds=time.time() - t0)
    return best


def fmt_entry(e: dict) -> str:
    status = '' if e['penalty'] == 0 else f"  PENALIZED (+{e['penalty']:.4f})"
    return (f"{describe_match(e['match'])}  peak_pressure={e['peak_pressure'] / 1e6:.2f}MPa"
            f"{status}\n      [{e['source']}] {sm.fmt_params(e['params'])}")


def plot_matches(picks, target_time, target_thrust, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(target_time, target_thrust, 'k-', lw=2.5, label='Target')
    for name, e in picks.items():
        ax.plot(e['time'], e['thrust'], lw=1.5,
                label=f"{name} (error {e['match']['match_error']:.3f})")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-csv', required=True,
                        help='CSV with the target thrust curve (columns: time, thrust)')
    parser.add_argument('--grains', nargs='+', default=list(sm.GRAINS), choices=list(sm.GRAINS))
    parser.add_argument('--max-pressure', type=float, default=None)
    parser.add_argument('--top', type=int, default=3, help='library designs to list per grain')
    parser.add_argument('--refine', type=int, default=1,
                        help='best eligible library designs per grain to refine (0 = none)')
    parser.add_argument('--refine-evals', type=int, default=40,
                        help='real simulations per refinement')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--plot', default=None, help='save target vs best-per-grain plot here')
    parser.add_argument('--save-json', default=None)
    add_structural_args(parser)
    args = parser.parse_args()
    case_spec = case_spec_from_args(args)

    target_time, target_thrust = read_target(args.target_csv)
    ref = thrust_match(target_time, target_thrust, target_time, target_thrust)
    print(f"Target {args.target_csv}: impulse {ref['target_impulse']:.0f}Ns, "
          f"burn time {ref['target_burn_time']:.2f}s, "
          f"average thrust {ref['target_avg_thrust']:.0f}N, peak {target_thrust.max():.0f}N")

    picks, eligible_by_grain, results = {}, {}, []
    for name in args.grains:
        t0 = time.time()
        ranked = screen(name, target_time, target_thrust, args.max_pressure, case_spec)
        eligible = eligible_by_grain[name] = [e for e in ranked if e['penalty'] == 0]
        print(f"\n=== {name}: {len(ranked)} library curves, {len(eligible)} eligible, "
              f"{time.time() - t0:.1f}s ===")
        for i, e in enumerate(eligible[:args.top], 1):
            print(f"  {i}. {fmt_entry(e)}")
        if eligible:
            picks[name] = eligible[0]
            results.append({'grain': name, 'stage': 'library', **eligible[0]})

    if args.refine > 0 and picks:
        jobs = [(name, e) for name in picks for e in eligible_by_grain[name][:args.refine]]
        print(f"\nRefining {len(jobs)} design(s) on the real simulator "
              f"({args.refine_evals} simulations each)...", flush=True)
        work = [partial(refine, e, name, target_time, target_thrust, args.max_pressure,
                        case_spec, args.refine_evals) for name, e in jobs]
        with multiprocessing.Pool(min(args.workers, len(work))) as pool:
            refined = pool.map(_call, work)
        for (name, start), r in zip(jobs, refined):
            if r is None:
                print(f"  {name}: refinement failed (every simulation raised)")
                continue
            gain = start['match']['match_error'] - r['match']['match_error']
            print(f"  {name}: {fmt_entry(r)}\n      {r['evals']} sims, {r['seconds']:.0f}s, "
                  f"error {start['match']['match_error']:.4f} -> {r['match']['match_error']:.4f}"
                  f" ({gain:+.4f})")
            results.append({'grain': name, 'stage': 'refined', **r})
            if r['penalty'] == 0 and r['match']['match_error'] < picks[name]['match']['match_error']:
                picks[name] = r

    if not picks:
        print("\nNo eligible design for any grain type.")
        return
    print("\n=== Best match per grain type (lower error is better) ===")
    for name, e in sorted(picks.items(), key=lambda kv: kv[1]['match']['match_error']):
        print(f"  {name:<8} {fmt_entry(e)}")
    best_name = min(picks, key=lambda n: picks[n]['match']['match_error'])
    print(f"\nBest grain type for this curve: {best_name}")

    if args.plot:
        plot_matches(picks, target_time, target_thrust, args.plot)
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, default=float)
        print(f"Saved results to {args.save_json}")


def _call(fn):
    return fn()


if __name__ == '__main__':
    main()
