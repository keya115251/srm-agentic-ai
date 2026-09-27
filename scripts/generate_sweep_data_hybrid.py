"""
BATES + Finocyl hybrid data generator: draws random two-segment motor
combos (Finocyl segment forward, BATES segment aft, one nozzle) and runs
each through simulator_bridge.hybrid_motor.run_hybrid_sim(), writing
(params, result) pairs to a CSV. Same approach and CSV layout as
generate_sweep_data_random.py (BATES) and generate_sweep_data_finocyl.py,
with one column per hybrid parameter.

Usage:
    PYTHONPATH=<path to openMotor>:<repo root> python scripts/generate_sweep_data_hybrid.py --n 1000 --workers 16

Sims are spread across worker processes. Rows are written as they finish
(order is not preserved) and flushed incrementally.
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator_bridge.hybrid_motor import run_hybrid_sim, HybridParams

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'hybrid_sweep_random.csv'
)

# --- Parameter bounds ---
# Starting from the BATES/Finocyl generators' bounds (two 0.05-0.09 m
# segments, throat 10-18 mm, exit/throat 1.5-3.5), only 1 of 50 pilot
# samples came out alert-free: 49 tripped the flow-separation warning. The
# hybrid's pressure curve is regressive (fin slivers and short, end-burning
# segments leave a long low-pressure tail), so exit pressure sits below
# 0.4 atm for more than 5% of the burn unless expansion is kept low. A
# 400-design scan over wider bounds showed exit/throat ratio is the main
# driver, with a longer BATES segment and a shorter Finocyl segment also
# helping; the bounds below keep ~59% of samples clean in a 120-sample
# pilot (BATES sweep: 276/500 = 55%).
DIAMETER_BOUNDS = (0.06, 0.10)                  # m, same as BATES
FINOCYL_LENGTH_BOUNDS = (0.05, 0.10)            # m, forward segment
BATES_LENGTH_BOUNDS = (0.10, 0.18)              # m, aft segment, same as BATES
# Wider than BATES: two segments burn more area, so 10-13 mm throats
# mostly tripped the max-pressure limit.
THROAT_BOUNDS = (0.013, 0.022)                  # m
EXIT_THROAT_RATIO_BOUNDS = (1.2, 2.0)           # multiple of throat
# Narrower than BATES's 0.30-0.75: wide aft cores left most of the burning
# area in the Finocyl segment and added to the separation tail.
BATES_CORE_FRAC_BOUNDS = (0.30, 0.60)           # of diameter
# Finocyl segment: same fin sampling scheme as the Finocyl generator.
FINOCYL_CORE_FRAC_BOUNDS = (0.25, 0.55)         # of diameter
NUM_FINS_BOUNDS = (3, 8)                        # inclusive integer range
FIN_LENGTH_FRAC_BOUNDS = (0.15, 0.70)           # of radial web outside the core
FIN_WIDTH_FRAC_BOUNDS = (0.15, 0.50)            # of core arc per fin

FEATURE_COLS = ['diameter', 'finocylLength', 'finocylCoreDiameter', 'numFins',
                'finWidth', 'finLength', 'batesLength', 'batesCoreDiameter',
                'throat', 'exit']
FIELDNAMES = FEATURE_COLS + [
    'success', 'burn_time', 'peak_thrust', 'total_impulse',
    'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
]


def sample_params(rng: random.Random) -> HybridParams:
    """Sample one random, geometrically valid hybrid combo. Dependent
    dimensions are sampled as fractions of the space available to them so
    cores can't exceed the OD and fins can't overrun the OD or overlap."""
    diameter = rng.uniform(*DIAMETER_BOUNDS)

    finocyl_length = rng.uniform(*FINOCYL_LENGTH_BOUNDS)
    finocyl_core = diameter * rng.uniform(*FINOCYL_CORE_FRAC_BOUNDS)
    num_fins = rng.randint(*NUM_FINS_BOUNDS)
    web = (diameter - finocyl_core) / 2
    fin_length = web * rng.uniform(*FIN_LENGTH_FRAC_BOUNDS)
    fin_width = (math.pi * finocyl_core / num_fins) * rng.uniform(*FIN_WIDTH_FRAC_BOUNDS)

    bates_length = rng.uniform(*BATES_LENGTH_BOUNDS)
    bates_core = diameter * rng.uniform(*BATES_CORE_FRAC_BOUNDS)

    throat = rng.uniform(*THROAT_BOUNDS)
    exit_dia = throat * rng.uniform(*EXIT_THROAT_RATIO_BOUNDS)

    return HybridParams(
        diameter=diameter, finocylLength=finocyl_length,
        finocylCoreDiameter=finocyl_core, numFins=num_fins,
        finWidth=fin_width, finLength=fin_length,
        batesLength=bates_length, batesCoreDiameter=bates_core,
        throat=throat, exit=exit_dia,
    )


def run_one(params: HybridParams) -> dict:
    row = dict(params)
    try:
        summary = run_hybrid_sim(params)
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
    print(f"Generating {args.n} random hybrid samples (seed={args.seed}, "
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
