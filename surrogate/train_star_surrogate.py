"""
Trains the Star surrogate: one GradientBoostingRegressor per output, as in
train_surrogate.py, on the 7 Star grain/nozzle inputs plus physics
features derived from them (see add_physics_features),
then runs the review_surrogate.py checks that matter for sign-off
(held-out R^2/MAE, 5-fold cross-validated R^2, feature importance).

Flagged samples are dropped before training. The one exception is the
"Low exit pressure, nozzle flow may separate" warning: motorlib raises it
when more than 5% of timesteps have exit pressure below 0.4 x ambient, and
every Star's sliver tail-off (points burnt through to the case, pressure
decaying for many timesteps) trips it regardless of the main-burn design,
as it does for Finocyl.
Dropping it would discard almost the whole dataset, so it is kept as a
column and reported, not used as a filter. Pass --drop-separation to treat
it like any other flag.

--holdout checks for overfitting: it scores the saved models on sims that
were generated separately (a different seed, never seen in training or the
test split) and prints train, test and fresh R^2 side by side.

Usage:
    python surrogate/train_star_surrogate.py --data data/raw/star_sweep_random.csv         --holdout data/raw/star_sweep_holdout.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Repo root on the path so the saved pipeline references
# surrogate.star_features (importable by the Design Agent), not __main__.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate.star_features import PHYSICS_FEATURES, add_physics_features

FEATURES = ['diameter', 'length', 'numPoints', 'pointLength',
            'pointWidth', 'throat', 'exit']
TARGETS = ['peak_thrust', 'total_impulse', 'peak_pressure', 'peak_kn']
SEPARATION_ALERT = 'Low exit pressure, nozzle flow may separate'
def load_data(path: str, drop_separation: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df['success'] == True].dropna(subset=TARGETS)
    after_fail = len(df)

    alerts = df['alerts'].fillna('').apply(
        lambda s: [a for a in s.split('; ') if a])
    ignored = set() if drop_separation else {SEPARATION_ALERT}
    flagged = alerts.apply(lambda a: any(x not in ignored for x in a))
    df = df[~flagged].copy()
    df['separation_flag'] = alerts[~flagged].apply(lambda a: SEPARATION_ALERT in a)

    print(f"Loaded {before} rows: {before - after_fail} failed sims, "
          f"{flagged.sum()} flagged ({'all alerts' if drop_separation else 'ignoring tail-off separation warning'}), "
          f"{len(df)} usable.")
    return df


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2,
                         random_state: int = 42):
    X = df[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    models, metrics = {}, {}

    for target in TARGETS:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        model = make_pipeline(
            FunctionTransformer(add_physics_features),
            GradientBoostingRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=5, random_state=random_state))
        model.fit(X_train, y_train)
        train_r2 = r2_score(y_train, model.predict(X_train))
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        cv = cross_val_score(clone(model), X, y, cv=kf, scoring='r2')

        metrics[target] = {
            'train_r2': train_r2, 'r2': r2_score(y_test, preds), 'mae': mae,
            'mean_actual': y_test.mean(),
            'pct_error': 100 * mae / y_test.mean(),
            'cv_scores': cv, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
            'importances': model[-1].feature_importances_,
        }
        models[target] = model

    return models, metrics


def print_report(metrics: dict):
    print("\n=== Held-out test set (20%) ===")
    print(f"{'Target':<16}{'R^2':>8}{'MAE':>16}{'Mean actual':>16}{'% error':>10}")
    for t, m in metrics.items():
        print(f"{t:<16}{m['r2']:>8.3f}{m['mae']:>16.1f}"
              f"{m['mean_actual']:>16.1f}{m['pct_error']:>9.1f}%")

    print("\n=== 5-fold cross-validated R^2 ===")
    for t, m in metrics.items():
        folds = ', '.join(f'{s:.3f}' for s in m['cv_scores'])
        print(f"  {t:<16} folds=[{folds}]  mean={m['cv_mean']:.3f}  std={m['cv_std']:.3f}")

    print("\n=== Feature importance per target ===")
    print("(top 5 of the raw + physics features)")
    names = FEATURES + PHYSICS_FEATURES
    for t, m in metrics.items():
        top = sorted(zip(names, m['importances']), key=lambda x: -x[1])[:5]
        print(f"  {t:<16}" + ', '.join(f"{n} {v:.3f}" for n, v in top))


def holdout_report(models: dict, metrics: dict, path: str, drop_separation: bool):
    """Scores the trained models on separately generated sims. A fresh R^2
    well below the test R^2 would mean the test split flattered the model."""
    df = load_data(path, drop_separation)
    X = df[FEATURES].values
    print(f"\n=== Fresh sims ({len(df)} usable rows from {path}) ===")
    print(f"{'Target':<16}{'train R^2':>10}{'test R^2':>10}{'fresh R^2':>11}{'fresh MAE':>14}{'% error':>9}")
    for t, model in models.items():
        y = df[t].values
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        print(f"{t:<16}{metrics[t]['train_r2']:>10.3f}{metrics[t]['r2']:>10.3f}"
              f"{r2_score(y, preds):>11.3f}{mae:>14.1f}{100 * mae / y.mean():>8.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/star_sweep_random.csv')
    parser.add_argument('--out', default='surrogate/star_surrogate_model.joblib')
    parser.add_argument('--drop-separation', action='store_true')
    parser.add_argument('--holdout', default=None,
                        help='CSV of separately generated sims to check for overfitting')
    args = parser.parse_args()

    df = load_data(args.data, args.drop_separation)
    if len(df) < 50:
        print(f"WARNING: only {len(df)} usable rows - results will be unreliable.")
    print(f"  {df['separation_flag'].mean():.0%} of kept rows carry the tail-off separation warning")

    models, metrics = train_and_evaluate(df)
    print_report(metrics)
    if args.holdout:
        holdout_report(models, metrics, args.holdout, args.drop_separation)

    # Training-data ranges for the raw inputs and the physics features, used
    # by star_surrogate.predict() to flag designs the model never saw.
    # Physics features are included because a design can sit inside every
    # raw range but still combine them into a Kn or volume nothing in
    # training reached.
    all_feats = add_physics_features(df[FEATURES].values)
    bounds = {name: (float(col.min()), float(col.max()))
              for name, col in zip(FEATURES + PHYSICS_FEATURES, all_feats.T)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'models': models, 'features': FEATURES, 'targets': TARGETS,
                 'bounds': bounds, 'grain_type': 'Star'}, args.out)
    print(f"\nSaved trained models to {args.out}")


if __name__ == '__main__':
    main()
