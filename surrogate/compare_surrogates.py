"""
Scores two or more saved surrogate models for the same grain type on one set
of held-out simulations, so models trained on different datasets (e.g. the
original random samples vs the Latin hypercube samples) are compared on
identical, unseen designs. Rows are filtered the same way that grain's
trainer filters its training data.

Usage:
    python surrogate/compare_surrogates.py --grain star \
        --holdout data/raw/star_sweep_lhs_holdout.csv \
        --models surrogate/star_surrogate_model.joblib surrogate/star_surrogate_model_lhs.joblib
"""
import argparse
import importlib
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINERS = {'bates': 'surrogate.train_surrogate',
            'finocyl': 'surrogate.train_finocyl_surrogate',
            'hybrid': 'surrogate.train_hybrid_surrogate',
            'star': 'surrogate.train_star_surrogate'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grain', required=True, choices=list(TRAINERS))
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()

    trainer = importlib.import_module(TRAINERS[args.grain])
    df = trainer.load_data(args.holdout)

    for path in args.models:
        d = joblib.load(path)
        X = df[d['features']].values
        print(f"\n=== {path} ===")
        print(f"{'Target':<16}{'R^2':>8}{'MAE':>14}{'% error':>9}{'p95 % err':>11}")
        for t, model in d['models'].items():
            y = df[t].values
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            p95 = np.percentile(100 * np.abs(pred - y) / np.abs(y), 95)
            print(f"{t:<16}{r2_score(y, pred):>8.4f}{mae:>14.1f}"
                  f"{100 * mae / y.mean():>8.2f}%{p95:>10.2f}%")


if __name__ == '__main__':
    main()
