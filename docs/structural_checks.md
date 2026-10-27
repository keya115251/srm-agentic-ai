# Structural checks: equations, verification and assumptions

Code: `structural_tools/casing.py`. Tests: `test/test_structural.py` (`python -m pytest test/`).

Every check uses the design's peak chamber pressure from the simulator (or the
surrogate during search) as the maximum expected operating pressure (MEOP), `p`.
The case inner radius is `r_i = D_grain / 2 + liner`, the outer radius
`r_o = r_i + t`. Margin of safety is `allowable / applied - 1`, so a negative
margin fails.

## Equations

| Check | Equation | Allowable |
|---|---|---|
| Case wall (governing) | Lame at the bore, closed ends: `s_hoop = p (r_o^2 + r_i^2) / (r_o^2 - r_i^2)`, `s_rad = -p`, `s_ax = p r_i^2 / (r_o^2 - r_i^2)`, combined by von Mises | `S_y / SF_y` |
| Case wall (reported only) | Thin wall: `s_hoop = p r_i / t`, `s_ax = p r_i / (2 t)` | n/a |
| Case burst | Barlow on the OD: `p_burst = 2 t S_u / D_o` | `p_burst >= p SF_u` |
| Forward bulkhead | Solid circular plate, uniform pressure (Roark Table 11.2): simply supported `3 (3 + nu) p a^2 / (8 h^2)` at the centre, clamped `3 p a^2 / (4 h^2)` at the edge, `a = r_i` | `S_y / SF_y` |
| Bolt shear | `F / (n pi d^2 / 4)`, `F = p pi r_i^2` | `S_y,bolt / sqrt(3) / SF_y` |
| Bolt bearing on the wall | `F / (n d t)` | `1.4 S_y / SF_y` |
| Wall tear-out | `F / (n 2 e t)` (two shear planes of length `e` per hole) | `S_y / sqrt(3) / SF_y` |
| Wall net tension at the bolt row | `F / (pi (r_o^2 - r_i^2) - n d t)` | `S_y / SF_y` |

The nozzle closure carries `p pi (r_i^2 - r_t^2)`, which is always less than the
bulkhead's `p pi r_i^2`, so the retention checks are sized on the bulkhead and a
pattern that passes also holds the nozzle.

## Verification

The structural equations were listed as unverified in the README. Each is
checked against the standard result it comes from:

1. **Lame satisfies equilibrium.** Integrating the Lame hoop stress through the
   wall gives exactly `p r_i` per unit length (the half-cylinder force balance),
   for walls from D/t = 100 down to D/t = 2. The axial stress times the wall
   area equals `p pi r_i^2` (end-cap balance). Radial stress is `-p` at the bore
   and 0 at the OD.
2. **Lame converges to thin wall.** The bore hoop stress exceeds `p r_i / t` by
   5.2% at D/t = 20, 1.0% at D/t = 100 and 0.1% at D/t = 1000. This is why the
   pass/fail uses Lame: the thin-wall formula is unconservative by that margin
   and is only reported.
3. **Plate coefficients match Roark.** `beta = 0.75` (clamped) and `1.2375`
   (simply supported, nu = 0.3) in `s = beta q a^2 / h^2`.
4. **Hand-worked example.** 98 mm grain, default case, 6 MPa: bore hoop
   95.7 MPa, bulkhead 111.6 MPa, bolt shear 133.4 MPa, bearing 198.0 MPa, all
   reproduced by the code; the design passes.

## Default hardware (placeholders, to be replaced with the mech team's)

| Item | Default |
|---|---|
| Case | 6061-T6 drawn tube, S_y 241 MPa, S_u 290 MPa (MMPDS minimums), 3.175 mm wall, no liner gap |
| Bulkhead | 6061-T6, 12.7 mm, simply supported (conservative) |
| Retention | 12 radial M6 class 8.8 bolts (S_y 640 MPa) per closure, 12 mm edge distance |
| Safety factors | 1.5 on yield, 2.0 on burst |

All of these are `CaseSpec` fields; the search scripts take `--case-wall-mm`,
`--bulkhead-mm`, `--n-bolts`, `--bolt-mm`, `--safety-factor`,
`--burst-safety-factor` and `--no-structural`.

## Where the checks run

- `design_agent/search_multigrain.py`: every candidate is checked at its
  predicted peak pressure and a negative margin is penalized like a pressure
  overshoot (5x the shortfall). The final pick is checked at its real peak
  pressure and is FLAGGED if it fails.
- `design_agent/kclass_sweep_multigrain.py`: same, and a failing design is not
  eligible for best match.
- `scripts/generate_sweep_data_lhs.py` writes `struct_ok`, `struct_min_margin`
  and `struct_failed` columns; `scripts/annotate_structural.py` adds them to
  existing sweeps without re-simulating. Rows are flagged, not dropped, since the
  surrogates model ballistics, which do not depend on the case.

With the default hardware, the share of the 5,000-sample LHS sweeps that pass:
BATES 90.0%, Hybrid 86.3%, Finocyl 72.0%, Star 71.7%. Bolt bearing is the most
common failure, then burst and bulkhead bending.

## Not covered

Thermal loads and strength derating at temperature, nozzle throat erosion,
snap-ring or threaded closures, O-ring groove stresses, fatigue, and buckling
under flight loads.
