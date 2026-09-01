"""
Phase 2 data generator (v2 - random sampling): draws random BATES grain +
nozzle parameter combos (instead of a fixed grid) and runs each through
simulator_bridge.bates_motor.run_bates_sim(), writing (params, result)
pairs to a CSV. Random sampling gives much better coverage of the
parameter space per simulation than a grid does - see project notes on
why 216 grid points wasn't enough.

Usage:
    PYTHONPATH=<path to openMotor> python scripts/generate_sweep_data.py --n 1500

Each sim takes ~5-6s (nozzle fsolve is the bottleneck), so:
    500 samples  ~= 45-50 min
    1000 samples ~= 90-100 min
    2000 samples ~= 3 hours

Run this in the background (or overnight) rather than blocking on it.
Progress is printed every 25 sims and the CSV is flushed incrementally,
so you can inspect partial results or kill/resume-append if needed.
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator_bridge.bates_motor import run_bates_sim, BatesParams

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'bates_sweep_random.csv'
)

# --- Parameter bounds (placeholders - tighten to mech team's real envelope) ---
DIAMETER_BOUNDS = (0.06, 0.10)         # m
LENGTH_BOUNDS = (0.10, 0.18)            # m
CORE_DIAMETER_FRAC_BOUNDS = (0.30, 0.75)  # coreDiameter as a fraction of diameter
THROAT_BOUNDS = (0.010, 0.018)          # m
# v1 used (1.5, 4.0) here and every single sample tripped mass-flux/pressure/
# Mach/exit-pressure alerts - the ratio was systematically too small for this
# diameter/throat range. Widened based on typical SRM area-expansion ratios
# (exit-to-throat DIAMETER ratio of ~3-9 covers area ratios of roughly 9-80,
# a reasonable range for low-altitude sea-level-optimized hobby motors).
EXIT_THROAT_RATIO_BOUNDS = (3.0, 9.0)   # exit as a multiple of throat


def sample_params(rng: random.Random) -> BatesParams:
    """Sample one random, physically-sane parameter combo.

    Core diameter is sampled as a fraction of the grain diameter (not an
    independent absolute value) so it can never accidentally exceed the
    grain diameter. Same idea for exit vs throat - keeps the random
    sampler from wasting sims on nonsensical geometry.
    """
    diameter = rng.uniform(*DIAMETER_BOUNDS)
    length = rng.uniform(*LENGTH_BOUNDS)
    core_diameter = diameter * rng.uniform(*CORE_DIAMETER_FRAC_BOUNDS)
    throat = rng.uniform(*THROAT_BOUNDS)
    exit_dia = throat * rng.uniform(*EXIT_THROAT_RATIO_BOUNDS)

    return BatesParams(
        diameter=diameter, length=length, coreDiameter=core_diameter,
        throat=throat, exit=exit_dia,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500,
                         help='number of random samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                         help='RNG seed for reproducibility')
    parser.add_argument('--append', action='store_true',
                         help='append to existing CSV instead of overwriting')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fieldnames = [
        'diameter', 'length', 'coreDiameter', 'throat', 'exit',
        'success', 'burn_time', 'peak_thrust', 'total_impulse',
        'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
    ]

    mode = 'a' if (args.append and os.path.exists(OUTPUT_PATH)) else 'w'
    write_header = mode == 'w'

    print(f"Generating {args.n} random samples (seed={args.seed}) "
          f"-> {OUTPUT_PATH} (mode={mode})")

    n_ok, n_fail = 0, 0
    t0 = time.time()

    with open(OUTPUT_PATH, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i in range(1, args.n + 1):
            params = sample_params(rng)
            row = dict(params)
            try:
                summary = run_bates_sim(params)
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
                n_ok += 1
            except Exception as e:
                row.update({
                    'success': False, 'burn_time': '', 'peak_thrust': '',
                    'total_impulse': '', 'peak_pressure': '', 'peak_kn': '',
                    'n_alerts': '', 'alerts': '', 'error': str(e),
                })
                n_fail += 1

            writer.writerow(row)
            f.flush()  # so partial progress survives an interrupt/kill

            if i % 25 == 0 or i == args.n:
                elapsed = time.time() - t0
                rate = elapsed / i
                remaining = rate * (args.n - i)
                print(f"  {i}/{args.n} done ({n_ok} ok, {n_fail} failed, "
                      f"{elapsed:.0f}s elapsed, ~{remaining/60:.1f} min left)")

    print(f"\nWrote {n_ok + n_fail} rows to {OUTPUT_PATH}")
    print(f"  {n_ok} successful sims, {n_fail} failed/errored")


if __name__ == '__main__':
    main()
