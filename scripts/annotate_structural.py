"""
Adds the structural columns that generate_sweep_data_lhs.py now writes
(struct_ok, struct_min_margin, struct_failed) to sweep CSVs generated before
it did, without re-simulating: the checks only need each row's geometry and
peak_pressure. Rewrites each CSV in place and prints the pass rate.

Usage:
    python scripts/annotate_structural.py data/raw/*_sweep_lhs.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_sweep_data_lhs import structural_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs='+')
    args = parser.parse_args()

    for path in args.csvs:
        df = pd.read_csv(path)
        ok = df['success'] == True
        cols = df[ok].apply(lambda r: structural_columns(r.to_dict(), r['peak_pressure']),
                            axis=1, result_type='expand')
        for c in cols:
            df[c] = cols[c].reindex(df.index)  # blank for failed sims
        df.to_csv(path, index=False)

        passed = cols['struct_ok'].astype(bool)
        print(f"{path}: {passed.sum()}/{ok.sum()} successful sims pass "
              f"({passed.mean():.1%}); failures by mode:")
        modes = cols.loc[~passed, 'struct_failed'].str.split('; ').explode().value_counts()
        for mode, n in modes.items():
            print(f"    {mode:<18} {n}")


if __name__ == '__main__':
    main()
