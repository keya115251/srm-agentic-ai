"""
Deeper review of the trained surrogate model, beyond the R^2/MAE numbers
train_surrogate.py already reports. Checks:

1. Feature importance per target - which of the 5 inputs actually drives
   each output? (sanity check: does this match physical intuition?)
2. Residual distribution - are errors roughly centered at zero, or does
   the model systematically over/under-predict in some region?
3. Error vs. distance-to-training-data - is accuracy worse for designs
   that are "unusual" relative to the training set? (directly relevant to
   the DE-vs-random-search finding: surrogate was less accurate for the
   design DE found near the pressure constraint boundary)
4. Cross-validated R^2 (not just a single train/test split) - checks
   whether the reported accuracy is stable or was a lucky split.

Usage:
    python surrogate/review_surrogate.py --data data/raw/bates_sweep_random.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

FEATURES = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']
TARGETS = ['peak_thrust', 'total_impulse', 'peak_pressure', 'peak_kn']


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['success'] == True].dropna(subset=TARGETS)
    return df


def feature_importance_report(models: dict):
    print("\n=== Feature importance per target ===")
    print("(GradientBoostingRegressor's built-in importance - higher means")
    print(" that input drives more of the model's prediction for this output)\n")
    header = f"{'':16}" + "".join(f"{f:>14}" for f in FEATURES)
    print(header)
    for target, model in models.items():
        importances = model.feature_importances_
        row = f"{target:16}" + "".join(f"{v:>14.3f}" for v in importances)
        print(row)
    print("\nSanity check against physical intuition:")
    print("- throat should dominate peak_pressure and peak_kn (Kn = burn area / throat area)")
    print("- diameter/length should dominate total_impulse (bigger grain = more propellant)")
    print("- if throat has near-zero importance for peak_pressure, something is off")


def residual_report(df: pd.DataFrame, models: dict):
    print("\n=== Residual distribution per target ===")
    X = df[FEATURES].values
    for target, model in models.items():
        y_true = df[target].values
        y_pred = model.predict(X)
        residuals = y_pred - y_true
        pct_residuals = 100 * residuals / y_true
        print(f"\n{target}:")
        print(f"  mean residual: {residuals.mean():.2f} "
              f"({'over' if residuals.mean() > 0 else 'under'}-predicting on average)")
        print(f"  std residual: {residuals.std():.2f}")
        print(f"  pct residual: mean={pct_residuals.mean():+.1f}%, "
              f"std={pct_residuals.std():.1f}%")
        print(f"  worst 3 errors (as % of true value): "
              f"{sorted(np.abs(pct_residuals))[-3:]}")


def cross_validation_report(df: pd.DataFrame, models: dict):
    print("\n=== 5-fold cross-validated R^2 (checks if train/test split was lucky) ===")
    X = df[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for target, model in models.items():
        y = df[target].values
        # re-fit fresh copies for CV so we're not leaking the already-fitted model
        from sklearn.base import clone
        fresh_model = clone(model)
        scores = cross_val_score(fresh_model, X, y, cv=kf, scoring='r2')
        print(f"  {target}: R^2 per fold = {[f'{s:.3f}' for s in scores]}, "
              f"mean={scores.mean():.3f}, std={scores.std():.3f}")
    print("\n  Large std across folds = accuracy is unstable / sensitive to which")
    print("  data points happen to be in train vs test. Small std = robust.")


def density_vs_error_report(df: pd.DataFrame, models: dict):
    print("\n=== Error vs. local training-data density ===")
    print("(checks whether accuracy degrades in sparser regions of the input space,")
    print(" e.g. near constraint boundaries where DE tends to push designs)\n")

    X = df[FEATURES].values
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize since units differ

    nn = NearestNeighbors(n_neighbors=6).fit(X_norm)  # 6 = self + 5 nearest
    distances, _ = nn.kneighbors(X_norm)
    local_density = distances[:, 1:].mean(axis=1)  # exclude self (distance 0)

    # split into "dense" (well-sampled) vs "sparse" (edge-of-data) halves
    median_density = np.median(local_density)
    dense_mask = local_density <= median_density
    sparse_mask = ~dense_mask

    for target, model in models.items():
        y_true = df[target].values
        y_pred = model.predict(X)
        pct_err = 100 * np.abs(y_pred - y_true) / y_true

        print(f"  {target}: dense-region avg error={pct_err[dense_mask].mean():.1f}%, "
              f"sparse-region avg error={pct_err[sparse_mask].mean():.1f}%")

    print("\n  If sparse-region error is notably higher, that confirms the surrogate")
    print("  is less trustworthy in under-sampled parts of the design space -")
    print("  exactly what generating more data near constraint boundaries would fix.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/bates_sweep_random.csv')
    parser.add_argument('--model', default='surrogate/surrogate_model.joblib')
    args = parser.parse_args()

    df = load_data(args.data)
    d = joblib.load(args.model)
    models = d['models']

    print(f"Loaded {len(df)} usable rows from {args.data}")

    feature_importance_report(models)
    residual_report(df, models)
    cross_validation_report(df, models)
    density_vs_error_report(df, models)


if __name__ == '__main__':
    main()