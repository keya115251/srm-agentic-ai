"""
Generates exploratory plots across the whole training dataset (the CSV
produced by scripts/generate_sweep_data_random.py). Useful for actually
seeing what the surrogate was trained on, rather than just trusting the
summary statistics from train_surrogate.py or review_surrogate.py.

Produces 3 PNG files:
1. distributions.png - histogram of each of the 5 inputs and 4 outputs,
   so you can see the actual shape of your sampled design space and
   performance range at a glance.
2. input_output_scatter.png - a 5x4 grid of scatter plots (each input vs
   each output), the fastest way to visually confirm the relationships
   review_surrogate.py's feature-importance numbers describe (e.g. does
   throat visibly drive peak_pressure when you actually look at it?).
3. coverage_map.png - 2D scatter of the two most physically important
   input pairs (diameter vs length, throat vs exit), colored by whether
   that row tripped any alerts - shows visually where your "safe" design
   space actually is, which is the same information the dense/sparse
   density check in review_surrogate.py computes numerically.

Usage:
    python surrogate/plot_training_data.py --data data/raw/bates_sweep_random.csv
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ['diameter', 'length', 'coreDiameter', 'throat', 'exit']
TARGETS = ['peak_thrust', 'total_impulse', 'peak_pressure', 'peak_kn']


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df['success'] == True].dropna(subset=TARGETS)


def plot_distributions(df: pd.DataFrame, out_dir: str):
    cols = FEATURES + TARGETS
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=30, color='#4c72b0', edgecolor='white')
        axes[i].set_title(col)
        axes[i].grid(alpha=0.3)
    for j in range(len(cols), len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'Distribution of inputs and outputs (n={len(df)} samples)')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'distributions.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_input_output_scatter(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(len(TARGETS), len(FEATURES), figsize=(16, 11))
    for i, target in enumerate(TARGETS):
        for j, feature in enumerate(FEATURES):
            ax = axes[i, j]
            ax.scatter(df[feature], df[target], s=6, alpha=0.35, color='#c44e52')
            if i == len(TARGETS) - 1:
                ax.set_xlabel(feature, fontsize=9)
            if j == 0:
                ax.set_ylabel(target, fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle('Every input vs every output (visual check against feature-importance numbers)')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'input_output_scatter.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_coverage_map(df: pd.DataFrame, out_dir: str):
    has_alerts = df['n_alerts'] > 0 if 'n_alerts' in df.columns else pd.Series([False]*len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(df.loc[~has_alerts, 'diameter']*1000, df.loc[~has_alerts, 'length']*1000,
                     s=10, alpha=0.5, color='#55a868', label='Clean (0 alerts)')
    axes[0].scatter(df.loc[has_alerts, 'diameter']*1000, df.loc[has_alerts, 'length']*1000,
                     s=10, alpha=0.5, color='#c44e52', label='Flagged (has alerts)')
    axes[0].set_xlabel('diameter (mm)')
    axes[0].set_ylabel('length (mm)')
    axes[0].set_title('Grain size coverage')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df.loc[~has_alerts, 'throat']*1000, df.loc[~has_alerts, 'exit']*1000,
                     s=10, alpha=0.5, color='#55a868', label='Clean (0 alerts)')
    axes[1].scatter(df.loc[has_alerts, 'throat']*1000, df.loc[has_alerts, 'exit']*1000,
                     s=10, alpha=0.5, color='#c44e52', label='Flagged (has alerts)')
    axes[1].set_xlabel('throat (mm)')
    axes[1].set_ylabel('exit (mm)')
    axes[1].set_title('Nozzle geometry coverage')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle('Where the training data actually sits in design space')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'coverage_map.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/bates_sweep_random.csv')
    parser.add_argument('--out-dir', default='data/raw/plots')
    args = parser.parse_args()

    df = load_data(args.data)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loaded {len(df)} usable rows from {args.data}")

    plot_distributions(df, args.out_dir)
    plot_input_output_scatter(df, args.out_dir)
    plot_coverage_map(df, args.out_dir)


if __name__ == '__main__':
    main()