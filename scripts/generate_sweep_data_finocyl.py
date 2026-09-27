"""
Finocyl data generator: draws random Finocyl grain + nozzle parameter combos
and runs each through simulator_bridge.finocyl_motor.run_finocyl_sim(),
writing (params, result) pairs to a CSV. Same approach and CSV layout as
generate_sweep_data_random.py (the BATES generator), plus the three fin
columns.

Usage:
    PYTHONPATH=<path to openMotor>:<repo root> python scripts/generate_sweep_data_finocyl.py --n 1000 --workers 16

Each sim takes ~5s, so sims are spread across worker processes. Rows are
written as they finish (order is not preserved) and flushed incrementally.
"""
import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator_bridge.finocyl_motor import run_finocyl_sim, FinocylParams

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'finocyl_sweep_random.csv'
)

# --- Parameter bounds ---
# Overall envelope and nozzle bounds match the BATES generator so the two
# datasets cover the same motor size range.
DIAMETER_BOUNDS = (0.06, 0.10)            # m
LENGTH_BOUNDS = (0.10, 0.18)              # m
THROAT_BOUNDS = (0.010, 0.018)            # m
EXIT_THROAT_RATIO_BOUNDS = (1.5, 3.5)     # multiple of throat, same as BATES
# Fins add burning area on top of the core, so the core is kept narrower
# than the BATES range to leave radial room for the fins plus a web.
CORE_DIAMETER_FRAC_BOUNDS = (0.25, 0.55)  # coreDiameter as a fraction of diameter
NUM_FINS_BOUNDS = (3, 8)                  # inclusive integer range
# finLength is measured outward from the core edge (motorlib adds it to the
# core radius), so it is sampled as a fraction of the radial web between
# core and grain OD. The upper bound leaves a web outside the fin tips.
FIN_LENGTH_FRAC_BOUNDS = (0.15, 0.70)
# finWidth is sampled as a fraction of the core arc per fin
# (pi * coreDiameter / numFins) so adjacent fins never merge at the core.
FIN_WIDTH_FRAC_BOUNDS = (0.15, 0.50)

FEATURE_COLS = ['diameter', 'length', 'coreDiameter', 'numFins',
                'finWidth', 'finLength', 'throat', 'exit']
FIELDNAMES = FEATURE_COLS + [
    'success', 'burn_time', 'peak_thrust', 'total_impulse',
    'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
]


def sample_params(rng: random.Random) -> FinocylParams:
    """Sample one random, geometrically valid Finocyl combo. Dependent
    dimensions are sampled as fractions of the space available to them so
    fins can't overrun the grain OD or overlap each other."""
    diameter = rng.uniform(*DIAMETER_BOUNDS)
    length = rng.uniform(*LENGTH_BOUNDS)
    core_diameter = diameter * rng.uniform(*CORE_DIAMETER_FRAC_BOUNDS)
    num_fins = rng.randint(*NUM_FINS_BOUNDS)
    web = (diameter - core_diameter) / 2
    fin_length = web * rng.uniform(*FIN_LENGTH_FRAC_BOUNDS)
    fin_width = (3.141592653589793 * core_diameter / num_fins) * rng.uniform(*FIN_WIDTH_FRAC_BOUNDS)
    throat = rng.uniform(*THROAT_BOUNDS)
    exit_dia = throat * rng.uniform(*EXIT_THROAT_RATIO_BOUNDS)

    return FinocylParams(
        diameter=diameter, length=length, coreDiameter=core_diameter,
        numFins=num_fins, finWidth=fin_width, finLength=fin_length,
        throat=throat, exit=exit_dia,
    )


def run_one(params: FinocylParams) -> dict:
    row = dict(params)
    try:
        summary = run_finocyl_sim(params)
        row.update({
            'success': summary['success'],
            'burn_time': summary['burn_time'],
            'peak_thrust': summary['peak_thrust'],
            'total_impulse': summary['total_impulse'],
            'peak_pressure': summary['peak_pressure'],
            'peak_kn': summary['peak_kn'],
            'n_alerts': len(summary['alerts']),
            'alerts': '; '.join(summary['alerts']),
            'error': '',
        })
    except Exception as e:
        row.update({
            'success': False, 'burn_time': '', 'peak_thrust': '',
            'total_impulse': '', 'peak_pressure': '', 'peak_kn': '',
            'n_alerts': '', 'alerts': '', 'error': str(e),
        })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000,
                         help='number of random samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                         help='RNG seed for reproducibility')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--out', default=OUTPUT_PATH)
    parser.add_argument('--append', action='store_true',
                         help='append to existing CSV instead of overwriting')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_params = [sample_params(rng) for _ in range(args.n)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    mode = 'a' if (args.append and os.path.exists(args.out)) else 'w'
    print(f"Generating {args.n} random Finocyl samples (seed={args.seed}, "
          f"workers={args.workers}) -> {args.out} (mode={mode})")

    n_ok, n_fail = 0, 0
    t0 = time.time()
    with open(args.out, mode, newline='') as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == 'w':
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, all_params), 1):
            writer.writerow(row)
            f.flush()
            if row['error']:
                n_fail += 1
            else:
                n_ok += 1
            if i % 50 == 0 or i == args.n:
                elapsed = time.time() - t0
                remaining = elapsed / i * (args.n - i)
                print(f"  {i}/{args.n} done ({n_ok} ok, {n_fail} failed, "
                      f"{elapsed:.0f}s elapsed, ~{remaining/60:.1f} min left)", flush=True)

    print(f"\nWrote {n_ok + n_fail} rows to {args.out}")
    print(f"  {n_ok} successful sims, {n_fail} failed/errored")


if __name__ == '__main__':
    main()
