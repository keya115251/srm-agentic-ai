"""
Star data generator: draws random Star grain + nozzle parameter combos and
runs each through simulator_bridge.star_motor.run_star_sim(), writing
(params, result) pairs to a CSV. Same approach and CSV layout as
generate_sweep_data_finocyl.py, with the three star-point columns in place
of the core and fin columns.

Usage:
    PYTHONPATH=<path to openMotor>:<repo root> python scripts/generate_sweep_data_star.py --n 1500 --workers 16

Each sim takes a few seconds, so sims are spread across worker processes.
Rows are written as they finish (order is not preserved) and flushed
incrementally.
"""
import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator_bridge.star_motor import run_star_sim, StarParams

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'star_sweep_random.csv'
)

# --- Parameter bounds ---
# Overall envelope and nozzle bounds match the BATES and Finocyl generators
# so all datasets cover the same motor size range.
DIAMETER_BOUNDS = (0.06, 0.10)            # m
LENGTH_BOUNDS = (0.10, 0.18)              # m
THROAT_BOUNDS = (0.010, 0.018)            # m
EXIT_THROAT_RATIO_BOUNDS = (1.5, 3.5)     # multiple of throat, same as BATES
NUM_POINTS_BOUNDS = (3, 8)                # inclusive integer range
# pointLength is sampled as a fraction of the grain radius; the upper bound
# leaves a web outside the point tips (motorlib warns past the radius).
# pointWidth is sampled as a fraction of pointLength, so a point's shape
# (slender vs stubby) is independent of its size.
# A star has no core circle, so its initial port is small. A 200-sample
# pilot over (0.25-0.85, 0.2-1.0) had 62% of designs flagged, mostly port/
# throat < 2 and mass-flux alerts from short, slender points; raising the
# lower bounds to these values cut that to ~30%, the rest mostly over-
# pressure from large burning areas.
POINT_LENGTH_FRAC_BOUNDS = (0.50, 0.90)
POINT_WIDTH_FRAC_BOUNDS = (0.40, 1.00)

FEATURE_COLS = ['diameter', 'length', 'numPoints', 'pointLength',
                'pointWidth', 'throat', 'exit']
FIELDNAMES = FEATURE_COLS + [
    'success', 'burn_time', 'peak_thrust', 'total_impulse',
    'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
]


def sample_params(rng: random.Random) -> StarParams:
    """Sample one random, geometrically valid Star combo. Dependent
    dimensions are sampled as fractions of the space available to them so
    the points can't overrun the grain OD."""
    diameter = rng.uniform(*DIAMETER_BOUNDS)
    length = rng.uniform(*LENGTH_BOUNDS)
    num_points = rng.randint(*NUM_POINTS_BOUNDS)
    point_length = diameter / 2 * rng.uniform(*POINT_LENGTH_FRAC_BOUNDS)
    point_width = point_length * rng.uniform(*POINT_WIDTH_FRAC_BOUNDS)
    throat = rng.uniform(*THROAT_BOUNDS)
    exit_dia = throat * rng.uniform(*EXIT_THROAT_RATIO_BOUNDS)

    return StarParams(
        diameter=diameter, length=length, numPoints=num_points,
        pointLength=point_length, pointWidth=point_width,
        throat=throat, exit=exit_dia,
    )


def run_one(params: StarParams) -> dict:
    row = dict(params)
    try:
        summary = run_star_sim(params)
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
    parser.add_argument('--n', type=int, default=1500,
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
    print(f"Generating {args.n} random Star samples (seed={args.seed}, "
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
