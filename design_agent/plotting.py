"""
Plotting helper for the Design Agent. Generates thrust and pressure curve
plots for a chosen design, saved as PNG files. Can optionally overlay a
target curve (from curve_matching.py's workflow) for visual comparison
alongside the cosine-similarity number.

Usage as a library (typical - call after search_v3_hybrid.py or
curve_matching.py picks/validates a design):
    from design_agent.plotting import plot_design_curves
    plot_design_curves(sim_result, title="Best design (target 1300 Ns)",
                        out_path="outputs/best_design.png")

    # with a target curve overlay:
    plot_design_curves(sim_result, target_time=t_arr, target_pressure=p_arr,
                        title="Design vs target", out_path="outputs/comparison.png")

Usage standalone (plot a single design by its params):
    python design_agent/plotting.py --params-json '{"diameter":0.08,...}' \
        --out outputs/design_curves.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_design_curves(sim_result: dict, target_time=None, target_pressure=None,
                        target_thrust=None, target_kn=None,
                        title: str = "Motor performance",
                        out_path: str = "design_curves.png"):
    """Single combined-axis chart matching OpenMotor's own GUI plot style:
    Kn, Pressure (bar), and Force (N) all drawn together on one shared axis
    with a legend, plus a motor-statistics readout underneath. Pressure is
    converted to bar (not Pa) specifically so its numeric range sits close
    to Force and Kn on the same axis, matching how OpenMotor's own plot
    (which uses psi) keeps all three curves visually comparable.

    sim_result is whatever run_bates_sim() returns: dict with 'time',
    'thrust', 'pressure', 'kn' lists plus summary stats.
    """
    time = np.array(sim_result['time'])
    thrust = np.array(sim_result['thrust'])
    pressure_bar = np.array(sim_result['pressure']) / 1e5  # Pa -> bar
    kn = np.array(sim_result.get('kn', []))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, kn, color='#1f77b4', linewidth=1.6, label='Kn')
    ax.plot(time, pressure_bar, color='#ff7f0e', linewidth=1.6, label='Pressure (bar)')
    ax.plot(time, thrust, color='#2ca02c', linewidth=1.6, label='Force (N)')

    if target_time is not None:
        if target_kn is not None:
            ax.plot(target_time, target_kn, color='#1f77b4', linewidth=1.2,
                    linestyle='--', alpha=0.6, label='Target Kn')
        if target_pressure is not None:
            ax.plot(target_time, np.array(target_pressure) / 1e5, color='#ff7f0e',
                    linewidth=1.2, linestyle='--', alpha=0.6, label='Target Pressure')
        if target_thrust is not None:
            ax.plot(target_time, target_thrust, color='#2ca02c', linewidth=1.2,
                    linestyle='--', alpha=0.6, label='Target Force')

    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Motor statistics readout underneath, matching OpenMotor's stats panel
    burn_time = sim_result.get('burn_time', time[-1] if len(time) else 0)
    peak_thrust = sim_result.get('peak_thrust', thrust.max() if len(thrust) else 0)
    total_impulse = sim_result.get('total_impulse', 0)
    peak_pressure_bar = sim_result.get('peak_pressure', 0) / 1e5
    avg_pressure_bar = pressure_bar.mean() if len(pressure_bar) else 0
    peak_kn = kn.max() if len(kn) else sim_result.get('peak_kn', 0)

    stats_lines = [
        f"Burn Time: {burn_time:.2f} s        Impulse: {total_impulse:.1f} Ns        "
        f"Peak Thrust: {peak_thrust:.1f} N",
        f"Average Pressure: {avg_pressure_bar:.1f} bar        "
        f"Peak Pressure: {peak_pressure_bar:.1f} bar        Peak Kn: {peak_kn:.1f}",
    ]
    fig.text(0.5, -0.04, "\n".join(stats_lines), ha='center', fontsize=9.5,
              color='#222', family='monospace')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-json', required=True)
    parser.add_argument('--out', default='design_curves.png')
    parser.add_argument('--title', default='Design performance')
    args = parser.parse_args()

    from simulator_bridge.bates_motor import run_bates_sim
    params = json.loads(args.params_json)
    result = run_bates_sim(params)
    plot_design_curves(result, title=args.title, out_path=args.out)


if __name__ == '__main__':
    main()