"""
Phase 2 data generator: sweeps BATES grain + nozzle parameters through
simulator_bridge.bates_motor.run_bates_sim() and writes each (params, result)
pair to a CSV. This CSV is the training set for the surrogate model in
surrogate/.

Usage:
    PYTHONPATH=. python scripts/generate_sweep_data.py

Ranges below are placeholders (roughly G/H-class hobby-motor scale, centered
on the mech team's known BATES config) - tighten these once you have the
mech team's actual design envelope / constraints.
"""
import csv
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator_bridge.bates_motor import run_bates_sim, BatesParams

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'bates_sweep.csv'
)

# --- Sweep ranges (placeholders - tighten to mech team's real envelope) ---
# NOTE: each sim takes ~5-6s (nozzle fsolve is the bottleneck). The full
# 3-value-per-param grid (~216 valid combos) takes ~20 min - fine to run
# yourself in a normal terminal, but too long for one command here. Using
# a smaller 2-value grid below to produce a working demo dataset now;
# widen back to 3+ values per param when you run this for real.
DIAMETER_RANGE = [0.06, 0.083058, 0.10]
LENGTH_RANGE = [0.10, 0.1397, 0.18]
CORE_DIAMETER_RANGE = [0.03, 0.05, 0.07]
THROAT_RANGE = [0.010, 0.01428, 0.018]
EXIT_RANGE = [0.035, 0.045, 0.055]

def core_diameter_valid(diameter: float, core_diameter: float) -> bool:
    """Basic sanity filter: core must leave meaningful web thickness."""
    return core_diameter < diameter * 0.85


def build_param_grid():
    combos = itertools.product(
        DIAMETER_RANGE, LENGTH_RANGE, CORE_DIAMETER_RANGE,
        THROAT_RANGE, EXIT_RANGE
    )
    for diameter, length, core_diameter, throat, exit_dia in combos:
        if not core_diameter_valid(diameter, core_diameter):
            continue
        if throat >= exit_dia:
            continue
        yield BatesParams(
            diameter=diameter, length=length, coreDiameter=core_diameter,
            throat=throat, exit=exit_dia,
        )


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    param_grid = list(build_param_grid())
    print(f"Running {len(param_grid)} simulations...")

    fieldnames = [
        'diameter', 'length', 'coreDiameter', 'throat', 'exit',
        'success', 'burn_time', 'peak_thrust', 'total_impulse',
        'peak_pressure', 'peak_kn', 'n_alerts', 'alerts', 'error',
    ]

    n_ok, n_fail = 0, 0
    t0 = time.time()

    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, params in enumerate(param_grid, 1):
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

            if i % 25 == 0 or i == len(param_grid):
                print(f"  {i}/{len(param_grid)} done "
                      f"({n_ok} ok, {n_fail} failed, "
                      f"{time.time() - t0:.1f}s elapsed)")

    print(f"\nWrote {n_ok + n_fail} rows to {OUTPUT_PATH}")
    print(f"  {n_ok} successful sims, {n_fail} failed/errored")


if __name__ == '__main__':
    main()
