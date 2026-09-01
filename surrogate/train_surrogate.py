"""
Phase 2: trains a surrogate model to approximate motorlib's simulation
output. Given the 5 grain/nozzle input parameters, predicts the 4 key
performance outputs in milliseconds instead of ~5-6s per real simulation.

This is what lets the Design Agent search many candidate geometries
quickly - it queries the surrogate during search, and only calls the
real simulator (simulator_bridge.bates_motor) to double-check the
final chosen design.

Usage:
    python surrogate/train_surrogate.py --data data/raw/bates_sweep_random.csv

Trains one GradientBoostingRegressor per output (thrust, impulse,
pressure, kn) since sklearn's GBM doesn't natively support multi-output
regression. Saves all 4 models + the feature list into one joblib file.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']
TARGETS = ['peak_thrust', 'total_impulse', 'peak_pressure', 'peak_kn']


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    # Drop failed/errored sims - can't train on rows with no output values
    df = df[df['success'] == True].dropna(subset=TARGETS)
    print(f"Loaded {before} rows, {len(df)} usable after dropping failures.")
    return df


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2,
                         random_state: int = 42) -> dict:
    X = df[FEATURES].values
    models = {}
    metrics = {}

    for target in TARGETS:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mean_actual = y_test.mean()
        pct_error = 100 * mae / mean_actual if mean_actual else float('nan')

        metrics[target] = {
            'mae': mae, 'r2': r2, 'pct_error': pct_error,
            'mean_actual': mean_actual,
        }
        models[target] = model

    return models, metrics


def print_report(metrics: dict):
    print("\n=== Surrogate model evaluation (held-out test set) ===")
    print(f"{'Target':<16}{'R^2':>8}{'MAE':>16}{'Mean actual':>16}{'% error':>10}")
    for target, m in metrics.items():
        print(f"{target:<16}{m['r2']:>8.3f}{m['mae']:>16.1f}"
              f"{m['mean_actual']:>16.1f}{m['pct_error']:>9.1f}%")
    print("\nR^2 close to 1.0 = good fit. % error is MAE as a fraction of the")
    print("average value being predicted - lower is better. Below ~10% is a")
    print("reasonable starting bar for a first-pass surrogate; the real")
    print("simulator should still be used to confirm final design choices.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/bates_sweep_random.csv')
    parser.add_argument('--out', default='surrogate/surrogate_model.joblib')
    args = parser.parse_args()

    df = load_data(args.data)
    if len(df) < 50:
        print(f"WARNING: only {len(df)} usable rows - results will be "
              f"unreliable. Consider generating more data first.")

    models, metrics = train_and_evaluate(df)
    print_report(metrics)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'models': models, 'features': FEATURES, 'targets': TARGETS},
                args.out)
    print(f"\nSaved trained models to {args.out}")


if __name__ == '__main__':
    main()