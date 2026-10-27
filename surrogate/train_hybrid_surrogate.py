"""
Trains the BATES + Finocyl hybrid surrogate: same model setup as
train_surrogate.py (one GradientBoostingRegressor per output) on the 10
hybrid grain/nozzle inputs, then runs the review_surrogate.py checks that
matter for sign-off (held-out R^2/MAE, 5-fold cross-validated R^2, feature
importance).

By default the 10 raw inputs are extended with physics features computed
from the geometry (initial burning area, Kn, propellant volume, port/throat
ratios, expansion ratio, webs; see surrogate/hybrid_features.py). They are
built inside the saved pipeline, so callers pass the same 10 raw inputs.
Pass --raw-only to train on the raw inputs alone.

Like the BATES surrogate, every flagged sample is dropped before training
(any safety alert, including the flow-separation warning), so the model
only learns the alert-free design space the Design Agent should search.
generate_sweep_data_hybrid.py's bounds were tuned so ~55% of samples
survive this filter, matching the BATES sweep.

Usage:
    python surrogate/train_hybrid_surrogate.py --data data/raw/hybrid_sweep_random.csv
"""
import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Import the feature helper as surrogate.hybrid_features (importable by the
# Design Agent) so the pickled pipeline doesn't reference __main__.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate.hybrid_features import PHYSICS_FEATURES, add_physics_features

FEATURES = ['diameter', 'finocylLength', 'finocylCoreDiameter', 'numFins',
            'finWidth', 'finLength', 'batesLength', 'batesCoreDiameter',
            'throat', 'exit']
TARGETS = ['peak_thrust', 'total_impulse', 'peak_pressure', 'peak_kn']


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df['success'] == True].dropna(subset=TARGETS)
    after_fail = len(df)
    df = df[df['n_alerts'] == 0]
    print(f"Loaded {before} rows: {before - after_fail} failed sims, "
          f"{after_fail - len(df)} flagged, {len(df)} usable.")
    return df


def train_and_evaluate(df: pd.DataFrame, physics: bool = True,
                         test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    models, metrics = {}, {}

    for target in TARGETS:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=random_state)
        model = (make_pipeline(FunctionTransformer(add_physics_features), gbm)
                 if physics else gbm)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        cv = cross_val_score(clone(model), X, y, cv=kf, scoring='r2')

        metrics[target] = {
            'r2': r2_score(y_test, preds), 'mae': mae,
            'mean_actual': y_test.mean(),
            'pct_error': 100 * mae / y_test.mean(),
            'cv_scores': cv, 'cv_mean': cv.mean(), 'cv_std': cv.std(),
            'importances': gbm.feature_importances_,
        }
        models[target] = model

    return models, metrics


def print_report(metrics: dict, feature_names: list):
    print("\n=== Held-out test set (20%) ===")
    print(f"{'Target':<16}{'R^2':>8}{'MAE':>16}{'Mean actual':>16}{'% error':>10}")
    for t, m in metrics.items():
        print(f"{t:<16}{m['r2']:>8.3f}{m['mae']:>16.1f}"
              f"{m['mean_actual']:>16.1f}{m['pct_error']:>9.1f}%")

    print("\n=== 5-fold cross-validated R^2 ===")
    for t, m in metrics.items():
        folds = ', '.join(f'{s:.3f}' for s in m['cv_scores'])
        print(f"  {t:<16} folds=[{folds}]  mean={m['cv_mean']:.3f}  std={m['cv_std']:.3f}")

    print("\n=== Feature importance per target (top 6) ===")
    for t, m in metrics.items():
        ranked = sorted(zip(feature_names, m['importances']), key=lambda x: -x[1])[:6]
        print(f"  {t:<16}" + ', '.join(f"{f} {v:.3f}" for f, v in ranked))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/hybrid_sweep_lhs.csv')
    parser.add_argument('--out', default='surrogate/hybrid_surrogate_model.joblib')
    parser.add_argument('--raw-only', action='store_true',
                        help='train on the 10 raw inputs without physics features')
    args = parser.parse_args()

    df = load_data(args.data)
    if len(df) < 50:
        print(f"WARNING: only {len(df)} usable rows - results will be unreliable.")

    physics = not args.raw_only
    models, metrics = train_and_evaluate(df, physics)
    print_report(metrics, FEATURES + (PHYSICS_FEATURES if physics else []))

    # Training-data ranges for the raw inputs and the physics features, used
    # by hybrid_surrogate.predict() to flag designs the model never saw.
    # Physics features are included because a design can sit inside every
    # raw range but still combine them into a Kn or volume nothing in
    # training reached. Stored even with --raw-only so the guard still works.
    all_feats = add_physics_features(df[FEATURES].values)
    bounds = {name: (float(col.min()), float(col.max()))
              for name, col in zip(FEATURES + PHYSICS_FEATURES, all_feats.T)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'models': models, 'features': FEATURES, 'targets': TARGETS,
                 'bounds': bounds, 'grain_type': 'Finocyl+BATES',
                 'physics_features': PHYSICS_FEATURES if physics else []}, args.out)
    print(f"\nSaved trained models to {args.out}")


if __name__ == '__main__':
    main()
