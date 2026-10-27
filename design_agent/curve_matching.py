"""
Curve matching for a chosen design against a target thrust (or pressure)
curve.

Two kinds of metric are reported:

  - thrust_match(): the scoring used to rank designs. Cosine similarity
    (Fan, Peng, Wei et al. 2026, Eq. 36) only compares curve *shape*: a
    design that delivers half the target's thrust for the same time scores
    a perfect 1.0, so on a K-class sweep every grain type tied near 0.99.
    thrust_match() compares absolute thrust instead. Its combined
    `match_error` (lower is better, 0 = identical curves) is

        nrmse + 0.5 * |impulse error| + 0.5 * |burn time error|

    where nrmse is the RMS thrust difference over the burn, divided by the
    target's average thrust, so it carries both level and shape, and the
    impulse and burn time errors are relative. The weights are fixed a
    priori and nothing is fitted to a particular target, so the same score
    applies to any target curve. Burn time is measured the same way on both
    curves (first to last time thrust is at or above 5% of its peak).
  - cosine similarity and relative impulse error (Fan et al. 2026, Eqs.
    35-36), still reported for comparison with the paper.

The surrogates only predict scalars (peak thrust, impulse, peak pressure,
peak Kn), so a design is matched to a whole curve either by running the real
simulator on it (compare_curves(), the K-class sweeps) or by screening the
full curves saved from the Latin hypercube runs (design_agent/match_curve.py).

Usage as a library:
    from design_agent.curve_matching import compare_curves, thrust_match
    result = compare_curves(candidate_params, target_time, target_thrust=target_thrust)
    print(result['thrust_match']['match_error'])

    m = thrust_match(sim['time'], sim['thrust'], target_time, target_thrust)

Usage standalone (a BATES design against a CSV of time,thrust and/or
time,pressure data):
    python design_agent/curve_matching.py --params-json '{"diameter":0.08,...}'         --target-csv path/to/target_curve.csv
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numpy 2.0 renamed trapz -> trapezoid; support both so this works regardless
# of which numpy version is installed.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def resample_to_grid(t_src, y_src, t_grid):
    """Linear-interpolates y_src(t_src) onto t_grid, zero-padding outside
    the source's time range (matches how a burned-out motor's pressure
    trails to zero, rather than extrapolating nonsense)."""
    return np.interp(t_grid, t_src, y_src, left=0.0, right=0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. 36 in Fan et al. (2026): normalized dot product of the two
    curves. 1.0 = identical shape, 0 = completely unrelated, can go
    negative for genuinely opposite curves (not expected here since
    pressure is always non-negative)."""
    num = np.sum(a * b)
    denom = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return num / denom if denom > 0 else 0.0


# Fraction of peak thrust that marks the start and end of the burn when
# measuring burn time. Applied identically to target and candidate, so a long
# low tail-off doesn't count as burn in one curve and not the other.
BURN_THRESHOLD = 0.05

# Weights of thrust_match()'s combined error. Fixed a priori, not fitted to
# any target curve.
MATCH_WEIGHTS = {'nrmse': 1.0, 'impulse': 0.5, 'burn_time': 0.5}


def burn_window(t, y, threshold: float = BURN_THRESHOLD):
    """(start, end) of the span where y is at or above threshold * max(y)."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if y.size == 0 or y.max() <= 0:
        return 0.0, 0.0
    above = np.flatnonzero(y >= threshold * y.max())
    return float(t[above[0]]), float(t[above[-1]])


def thrust_match(cand_time, cand_thrust, target_time, target_thrust,
                 dt: float = 0.005, weights: dict = None) -> dict:
    """Absolute thrust-curve comparison of a candidate against a target.
    Both curves are resampled onto one uniform grid from t=0 (both start at
    ignition), then:

      nrmse              RMS thrust difference over the union of both burn
                         windows, divided by the target's average thrust
      impulse_error      (I_cand - I_target) / I_target
      burn_time_error    (tb_cand - tb_target) / tb_target
      avg_thrust_error   same, for impulse / burn time
      peak_thrust_error  same, for the peak
      cosine_similarity  shape-only, for reference
      match_error        weighted sum of nrmse and the absolute impulse and
                         burn time errors (MATCH_WEIGHTS); lower is better

    Signed errors are positive when the candidate is above the target."""
    w = {**MATCH_WEIGHTS, **(weights or {})}
    cand_time = np.asarray(cand_time, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    grid = np.arange(0, max(cand_time[-1], target_time[-1]) + dt, dt)
    c = resample_to_grid(cand_time, np.asarray(cand_thrust, dtype=float), grid)
    g = resample_to_grid(target_time, np.asarray(target_thrust, dtype=float), grid)

    i_c, i_t = _trapz(c, grid), _trapz(g, grid)
    c0, c1 = burn_window(grid, c)
    g0, g1 = burn_window(grid, g)
    tb_c, tb_t = c1 - c0, g1 - g0
    avg_t = i_t / tb_t
    avg_c = i_c / tb_c if tb_c > 0 else 0.0
    in_burn = (grid >= min(c0, g0)) & (grid <= max(c1, g1))
    nrmse = float(np.sqrt(np.mean((c[in_burn] - g[in_burn]) ** 2)) / avg_t)

    out = {
        'nrmse': nrmse,
        'impulse_error': float((i_c - i_t) / i_t),
        'burn_time_error': float((tb_c - tb_t) / tb_t),
        'avg_thrust_error': float((avg_c - avg_t) / avg_t),
        'peak_thrust_error': float((c.max() - g.max()) / g.max()),
        'cosine_similarity': float(cosine_similarity(c, g)),
        'impulse': float(i_c), 'burn_time': float(tb_c), 'avg_thrust': float(avg_c),
        'target_impulse': float(i_t), 'target_burn_time': float(tb_t),
        'target_avg_thrust': float(avg_t),
    }
    out['match_error'] = (w['nrmse'] * nrmse + w['impulse'] * abs(out['impulse_error'])
                          + w['burn_time'] * abs(out['burn_time_error']))
    return out


def describe_match(m: dict) -> str:
    return (f"match_error={m['match_error']:.4f}  nrmse={m['nrmse']:.3f}  "
            f"impulse {m['impulse_error']:+.1%}  burn time {m['burn_time_error']:+.1%}  "
            f"avg thrust {m['avg_thrust_error']:+.1%}  peak {m['peak_thrust_error']:+.1%}  "
            f"(cos_sim {m['cosine_similarity']:.4f})")


def relative_impulse_error(t_a, p_a, t_b, p_b) -> float:
    """Eq. 35 in Fan et al. (2026): |I_d - I_t| / |I_t| as a percentage,
    where I is total impulse (trapezoidal integral of pressure... note the
    paper integrates chamber pressure directly as a proxy for impulse
    trend; for actual Ns-impulse comparison we integrate thrust instead if
    available - see compare_curves()."""
    I_a = _trapz(p_a, t_a)
    I_b = _trapz(p_b, t_b)
    return 100.0 * abs(I_a - I_b) / abs(I_b) if I_b != 0 else float('inf')


def compare_curves(candidate_params: dict, target_time: np.ndarray,
                    target_pressure: np.ndarray = None,
                    target_thrust: np.ndarray = None,
                    dt: float = 0.02) -> dict:
    """Runs the real simulator on candidate_params, resamples both curves
    onto a shared uniform time grid (matching the paper's approach of a
    fixed dt grid rather than comparing raw irregular timesteps), and
    reports cosine similarity + relative impulse error, plus, for a thrust
    target, thrust_match()'s absolute comparison under 'thrust_match'.

    Provide target_pressure and/or target_thrust depending on what your
    target data actually has (test-fire telemetry usually has pressure,
    a datasheet thrust curve would have thrust - the paper's Eq 35/36 use
    chamber pressure specifically, so pressure is preferred if you have it).
    """
    from simulator_bridge.bates_motor import run_bates_sim

    result = run_bates_sim(candidate_params)
    cand_time = np.array(result['time'])

    t_max = max(target_time[-1], cand_time[-1] if len(cand_time) else 0)
    t_grid = np.arange(0, t_max + dt, dt)

    out = {}

    if target_pressure is not None:
        cand_pressure = np.array(result['pressure'])
        target_p_grid = resample_to_grid(target_time, target_pressure, t_grid)
        cand_p_grid = resample_to_grid(cand_time, cand_pressure, t_grid)
        out['pressure_cosine_similarity'] = cosine_similarity(cand_p_grid, target_p_grid)
        out['pressure_relative_error_pct'] = relative_impulse_error(
            t_grid, cand_p_grid, t_grid, target_p_grid)

    if target_thrust is not None:
        cand_thrust = np.array(result['thrust'])
        target_t_grid = resample_to_grid(target_time, target_thrust, t_grid)
        cand_t_grid = resample_to_grid(cand_time, cand_thrust, t_grid)
        out['thrust_cosine_similarity'] = cosine_similarity(cand_t_grid, target_t_grid)
        # this one is genuine Ns impulse, since thrust integrates to impulse directly
        out['thrust_relative_impulse_error_pct'] = relative_impulse_error(
            t_grid, cand_t_grid, t_grid, target_t_grid)
        out['thrust_match'] = thrust_match(cand_time, cand_thrust, target_time, target_thrust)

    out['candidate_sim_result'] = result
    return out


def print_report(out: dict):
    print("=== Curve-shape validation ===")
    if 'pressure_cosine_similarity' in out:
        print(f"Pressure curve: cosine similarity={out['pressure_cosine_similarity']:.4f} "
              f"(1.0 = identical shape), relative error={out['pressure_relative_error_pct']:.2f}%")
    if 'thrust_cosine_similarity' in out:
        print(f"Thrust curve:   cosine similarity={out['thrust_cosine_similarity']:.4f} "
              f"(1.0 = identical shape), relative impulse error={out['thrust_relative_impulse_error_pct']:.2f}%")
    if 'thrust_match' in out:
        print(f"Thrust match:   {describe_match(out['thrust_match'])}")
    print("\nGuideline from the literature (Fan et al. 2026): cosine similarity")
    print("above ~0.98 with impulse error below ~2% indicates strong agreement")
    print("between the designed and target curves.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-json',
                         help='JSON dict of candidate design params, e.g. '
                              '\'{"diameter":0.08,"length":0.17,"coreDiameter":0.05,'
                              '"throat":0.014,"exit":0.079}\'. On Windows/PowerShell, '
                              'quoting this inline is unreliable - use --params-file instead.')
    parser.add_argument('--params-file',
                         help='Path to a JSON file containing the params dict. '
                              'Easier than --params-json on Windows/PowerShell, which '
                              'often mangles nested quotes when calling external programs.')
    parser.add_argument('--target-csv', required=True,
                         help='CSV with columns: time, pressure (and/or thrust)')
    args = parser.parse_args()

    if not args.params_json and not args.params_file:
        parser.error('provide either --params-json or --params-file')

    import csv
    time_vals, pressure_vals, thrust_vals = [], [], []
    with open(args.target_csv) as f:
        reader = csv.DictReader(f)
        has_pressure = 'pressure' in reader.fieldnames
        has_thrust = 'thrust' in reader.fieldnames
        for row in reader:
            time_vals.append(float(row['time']))
            if has_pressure:
                pressure_vals.append(float(row['pressure']))
            if has_thrust:
                thrust_vals.append(float(row['thrust']))

    if args.params_file:
        # utf-8-sig tolerates a BOM (byte-order-mark) at the start of the file,
        # which PowerShell's `-Encoding utf8` silently adds and which a plain
        # 'utf-8' open() would choke on with "Expecting value: line 1 column 1".
        with open(args.params_file, encoding='utf-8-sig') as f:
            params = json.load(f)
    else:
        params = json.loads(args.params_json)
    target_time = np.array(time_vals)
    target_pressure = np.array(pressure_vals) if pressure_vals else None
    target_thrust = np.array(thrust_vals) if thrust_vals else None

    out = compare_curves(params, target_time, target_pressure, target_thrust)
    print_report(out)


if __name__ == '__main__':
    main()