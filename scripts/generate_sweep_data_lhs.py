"""
Latin hypercube data generator for any of the four grain types. Replaces
independent random draws (generate_sweep_data_random.py and the per-grain
generators) with a Latin hypercube design: every input dimension is split
into n equal strata and each stratum is sampled exactly once, so the design
space is covered evenly instead of leaving the clumps and gaps that random
sampling leaves at the same sample count.

The search space is the same one the existing generators and
design_agent/search_multigrain.py use - each grain's bounds and decoder are
taken from search_multigrain.GRAINS, so dependent dimensions (core diameter,
fin length, point length...) are still sampled as fractions of the room
available to them, and integer dimensions (numFins, numPoints) get an equal
share of the unit interval each.

Besides the usual summary CSV (same columns as the other generators, plus
sample_id), each simulation's full time/thrust/pressure/Kn curve is written to
a sidecar JSON-lines file keyed by sample_id. The surrogates only use the
scalars today, but a surrogate that predicts whole curves (needed to fit an
arbitrary target curve) can be trained from the same runs without
re-simulating.

Usage:
    PYTHONPATH=<path to openMotor>:<repo root> python scripts/generate_sweep_data_lhs.py --grain star --n 5000
    # -> data/raw/star_sweep_lhs.csv and data/raw/star_sweep_lhs_curves.jsonl
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from design_agent.search_multigrain import GRAINS
from structural_tools.casing import check_structure

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'raw')
GRAIN_NAMES = {name.lower(): name for name in GRAINS}
OUTPUTS = ['success', 'burn_time', 'peak_thrust', 'total_impulse',
           'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
           'struct_ok', 'struct_min_margin', 'struct_failed']


def structural_columns(params: dict, peak_pressure: float) -> dict:
    """Structural verdict for the default case hardware (see
    structural_tools/casing.py). Recorded as a flag, not used to drop rows:
    the surrogates model ballistics, which don't depend on the case."""
    st = check_structure(params, peak_pressure)
    return {'struct_ok': st['ok'], 'struct_min_margin': round(st['min_margin'], 4),
            'struct_failed': '; '.join(st['failed'])}


def lhs_params(grain_name: str, n: int, seed: int) -> list:
    """n Latin hypercube samples, decoded into wrapper params. random-cd
    optimization shuffles the strata to lower the design's discrepancy, so
    the points are also spread well in 2-D projections, not only per axis."""
    grain = GRAINS[grain_name]
    lo, hi = np.array(grain.bounds).T
    unit = qmc.LatinHypercube(d=len(lo), optimization='random-cd', seed=seed).random(n)
    return [grain.decode(lo + u * (hi - lo)) for u in unit]


def run_one(job):
    grain_name, sample_id, params = job
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    row = {'sample_id': sample_id, **params}
    curve = None
    try:
        s = GRAINS[grain_name].simulate(params)
        row.update({
            'success': s['success'], 'burn_time': s['burn_time'],
            'peak_thrust': s['peak_thrust'], 'total_impulse': s['total_impulse'],
            'peak_pressure': s['peak_pressure'], 'peak_kn': s['peak_kn'],
            'n_alerts': len(s['alerts']), 'alerts': '; '.join(s['alerts']), 'error': '',
            **structural_columns(params, s['peak_pressure']),
        })
        curve = {'sample_id': sample_id,
                 **{k: [round(v, 6) for v in s[k]]
                    for k in ('time', 'thrust', 'pressure', 'kn')}}
    except Exception as e:
        row.update({k: '' for k in OUTPUTS})
        row.update({'success': False, 'error': str(e)})
    return row, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grain', required=True, choices=list(GRAIN_NAMES))
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--out', default=None,
                        help='CSV path (default data/raw/<grain>_sweep_lhs.csv); the '
                             'curves go next to it with a _curves.jsonl suffix')
    args = parser.parse_args()

    grain_name = GRAIN_NAMES[args.grain]
    out = args.out or os.path.join(RAW_DIR, f'{args.grain}_sweep_lhs.csv')
    curves_out = os.path.splitext(out)[0] + '_curves.jsonl'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    all_params = lhs_params(grain_name, args.n, args.seed)
    fieldnames = ['sample_id'] + list(all_params[0]) + OUTPUTS
    jobs = [(grain_name, i, p) for i, p in enumerate(all_params)]
    print(f"Generating {args.n} Latin hypercube {grain_name} samples (seed={args.seed}, "
          f"workers={args.workers}) -> {out}", flush=True)

    n_ok, n_fail = 0, 0
    t0 = time.time()
    with open(out, 'w', newline='') as f, open(curves_out, 'w') as fc, \
            Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (row, curve) in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
            writer.writerow(row)
            if curve is not None:
                fc.write(json.dumps(curve) + '\n')
                n_ok += 1
            else:
                n_fail += 1
            if i % 250 == 0 or i == args.n:
                f.flush()
                fc.flush()
                elapsed = time.time() - t0
                print(f"  {i}/{args.n} done ({n_ok} ok, {n_fail} failed, {elapsed:.0f}s elapsed, "
                      f"~{elapsed / i * (args.n - i) / 60:.1f} min left)", flush=True)

    print(f"\nWrote {n_ok + n_fail} rows to {out} and {n_ok} curves to {curves_out}")


if __name__ == '__main__':
    main()
