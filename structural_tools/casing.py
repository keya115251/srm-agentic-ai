"""
Structural feasibility checks for the motor case and its end closures, so a
design whose chamber pressure the hardware cannot hold is flagged instead of
being treated as a valid pick. Every check takes the design's grain/nozzle
params (the same dicts the simulator_bridge wrappers take) plus its peak
chamber pressure, which is used as the maximum expected operating pressure
(MEOP), and returns stress, allowable and margin for each failure mode:

    from structural_tools.casing import check_structure
    s = check_structure(params, peak_pressure)
    s['ok'], s['min_margin'], s['failed']

Failure modes checked (equations and their verification are written up in
docs/structural_checks.md, and test/test_structural.py checks them):
  - case wall: Lame thick-wall stresses at the bore, closed ends, combined
    by von Mises, against yield / SF_yield. The thin-wall hoop stress p*r/t
    is reported alongside it, but not used to pass or fail, because it
    underestimates the bore hoop stress by ~5% at D/t = 20 and more for
    thicker walls.
  - case burst: Barlow burst pressure 2*t*S_u/D_o against MEOP * SF_ult.
  - forward bulkhead: flat circular plate under uniform pressure, max
    bending stress (Roark), against yield / SF_yield.
  - closure retention (radial bolts through the case wall): bolt shear,
    bearing on the case wall, shear tear-out of the case wall behind each
    bolt, and net-section tension of the case at the bolt row. Sized for the
    forward bulkhead's axial load p*pi*r_i^2, which is always larger than the
    nozzle closure's p*pi*(r_i^2 - r_t^2), so a bolt pattern that passes here
    also holds the nozzle.

The case inner diameter is the grain diameter plus twice the liner
thickness; the casing hardware itself (wall, bulkhead, bolts, materials) is
not in the design vector yet, so it comes from a CaseSpec. The defaults
below are placeholders for a typical 6061-T6 amateur case and should be
replaced with the mech team's actual hardware.
"""
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Material:
    name: str
    yield_strength: float      # Pa
    ultimate_strength: float   # Pa
    poisson: float = 0.33


# MMPDS minimums for 6061-T6 drawn tube (Fty 35 ksi, Ftu 42 ksi).
AL_6061_T6 = Material('6061-T6 aluminium', 241e6, 290e6, 0.33)
# ISO 898-1 property class 8.8 (d <= 16 mm), minimum yield and tensile.
STEEL_8_8 = Material('Class 8.8 steel bolt', 640e6, 800e6, 0.29)

# Von Mises shear yield, as a fraction of tensile yield (1/sqrt(3)).
SHEAR_YIELD_RATIO = 1 / math.sqrt(3)
# Bearing yield allowable, as a fraction of tensile yield. MMPDS gives
# Fbry ~1.43 Fty for 6061-T6 at edge distance / diameter = 1.5; 1.4 keeps it
# conservative for any e/D >= 1.5.
BEARING_YIELD_RATIO = 1.4


@dataclass(frozen=True)
class CaseSpec:
    wall_thickness: float = 3.175e-3       # m (1/8 in tube wall)
    liner_thickness: float = 0.0           # m, radial gap between grain OD and case ID
    bulkhead_thickness: float = 12.7e-3    # m (1/2 in plate)
    bulkhead_edge: str = 'simply_supported'  # or 'clamped'; simply supported is conservative
    n_bolts: int = 12                      # radial bolts per closure
    bolt_diameter: float = 6e-3            # m, diameter in the shear plane (shank)
    bolt_edge_distance: float = 12e-3      # m, bolt centre to the end of the case
    case_material: Material = field(default=AL_6061_T6)
    closure_material: Material = field(default=AL_6061_T6)
    bolt_material: Material = field(default=STEEL_8_8)
    safety_factor_yield: float = 1.5
    safety_factor_ultimate: float = 2.0


DEFAULT_CASE = CaseSpec()


# --- Stress equations ---

def thin_wall_stresses(p: float, r_i: float, t: float) -> dict:
    """Thin-walled closed-end cylinder (valid for t/r <~ 0.1): hoop from a
    half-cylinder force balance, axial from an end-cap force balance."""
    return {'hoop': p * r_i / t, 'axial': p * r_i / (2 * t)}


def lame_stresses(p: float, r_i: float, r_o: float, r: float = None) -> dict:
    """Lame thick-walled cylinder, internal pressure p, zero external
    pressure, closed ends, at radius r (default the bore, where hoop and
    radial stress are largest in magnitude):
        hoop   = A + B / r^2,  radial = A - B / r^2,  axial = A
        A = p r_i^2 / (r_o^2 - r_i^2),  B = p r_i^2 r_o^2 / (r_o^2 - r_i^2)
    """
    r = r_i if r is None else r
    a = p * r_i ** 2 / (r_o ** 2 - r_i ** 2)
    b = a * r_o ** 2
    return {'hoop': a + b / r ** 2, 'radial': a - b / r ** 2, 'axial': a}


def von_mises(s1: float, s2: float, s3: float) -> float:
    return math.sqrt(((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2) / 2)


def barlow_burst_pressure(t: float, d_o: float, ultimate: float) -> float:
    """Barlow's formula on the outside diameter, the conservative form used
    for pipe/tube burst ratings."""
    return 2 * t * ultimate / d_o


def flat_plate_max_stress(p: float, a: float, t: float, poisson: float,
                          edge: str = 'simply_supported') -> float:
    """Max bending stress in a solid circular plate of radius a and
    thickness t under uniform pressure p (Roark, Table 11.2):
        clamped edge:          3 p a^2 / (4 t^2)            (at the edge)
        simply supported edge: 3 (3 + nu) p a^2 / (8 t^2)   (at the centre)
    """
    if edge == 'clamped':
        return 3 * p * a ** 2 / (4 * t ** 2)
    if edge == 'simply_supported':
        return 3 * (3 + poisson) * p * a ** 2 / (8 * t ** 2)
    raise ValueError(f"unknown edge condition {edge!r}")


# --- Feasibility check ---

def _entry(stress: float, allowable: float) -> dict:
    # margin of safety = allowable / applied - 1; negative means it fails.
    margin = allowable / stress - 1 if stress > 0 else math.inf
    return {'stress': stress, 'allowable': allowable, 'margin': margin}


def check_structure(params: dict, peak_pressure: float,
                    spec: CaseSpec = DEFAULT_CASE) -> dict:
    """Checks a design's case and closures at MEOP = peak_pressure (Pa).
    Returns {'ok', 'min_margin', 'failed', 'checks', 'thin_wall_valid',
    'case_inner_diameter'}; each entry of 'checks' holds stress, allowable
    and margin in Pa (pressures for 'case_burst')."""
    p = max(float(peak_pressure), 0.0)
    r_i = params['diameter'] / 2 + spec.liner_thickness
    t = spec.wall_thickness
    r_o = r_i + t
    case, closure, bolt = spec.case_material, spec.closure_material, spec.bolt_material
    sf_y, sf_u = spec.safety_factor_yield, spec.safety_factor_ultimate
    checks = {}

    lame = lame_stresses(p, r_i, r_o)
    checks['case_von_mises'] = _entry(von_mises(lame['hoop'], lame['radial'], lame['axial']),
                                      case.yield_strength / sf_y)
    # Applied side is MEOP * SF_ult, allowable is the burst pressure.
    checks['case_burst'] = _entry(p * sf_u, barlow_burst_pressure(t, 2 * r_o, case.ultimate_strength))

    checks['bulkhead_bending'] = _entry(
        flat_plate_max_stress(p, r_i, spec.bulkhead_thickness, closure.poisson, spec.bulkhead_edge),
        closure.yield_strength / sf_y)

    force = p * math.pi * r_i ** 2
    n, d, e = spec.n_bolts, spec.bolt_diameter, spec.bolt_edge_distance
    checks['bolt_shear'] = _entry(force / (n * math.pi * d ** 2 / 4),
                                  SHEAR_YIELD_RATIO * bolt.yield_strength / sf_y)
    checks['bolt_bearing'] = _entry(force / (n * d * t),
                                    BEARING_YIELD_RATIO * case.yield_strength / sf_y)
    # Each bolt pushes out a plug of wall along two shear planes of length
    # e (edge distance) back to the end of the case.
    checks['wall_tearout'] = _entry(force / (n * 2 * e * t),
                                    SHEAR_YIELD_RATIO * case.yield_strength / sf_y)
    net_area = math.pi * (r_o ** 2 - r_i ** 2) - n * d * t
    checks['wall_net_tension'] = _entry(force / net_area if net_area > 0 else math.inf,
                                        case.yield_strength / sf_y)

    failed = [name for name, c in checks.items() if c['margin'] < 0]
    return {
        'ok': not failed,
        'min_margin': min(c['margin'] for c in checks.values()),
        'failed': failed,
        'checks': checks,
        'thin_wall_hoop': thin_wall_stresses(p, r_i, t)['hoop'],
        'thin_wall_valid': t / r_i <= 0.1,
        'case_inner_diameter': 2 * r_i,
    }


# --- CLI helpers shared by the search scripts ---

def add_structural_args(parser):
    g = parser.add_argument_group('structural checks (see structural_tools/casing.py)')
    g.add_argument('--case-wall-mm', type=float, default=DEFAULT_CASE.wall_thickness * 1e3)
    g.add_argument('--bulkhead-mm', type=float, default=DEFAULT_CASE.bulkhead_thickness * 1e3)
    g.add_argument('--n-bolts', type=int, default=DEFAULT_CASE.n_bolts)
    g.add_argument('--bolt-mm', type=float, default=DEFAULT_CASE.bolt_diameter * 1e3)
    g.add_argument('--safety-factor', type=float, default=DEFAULT_CASE.safety_factor_yield,
                   help='safety factor on yield (burst uses --burst-safety-factor)')
    g.add_argument('--burst-safety-factor', type=float,
                   default=DEFAULT_CASE.safety_factor_ultimate)
    g.add_argument('--no-structural', action='store_true',
                   help='skip the structural checks entirely')


def case_spec_from_args(args):
    """CaseSpec from add_structural_args() options, or None with --no-structural."""
    if getattr(args, 'no_structural', False):
        return None
    return CaseSpec(wall_thickness=args.case_wall_mm / 1e3,
                    bulkhead_thickness=args.bulkhead_mm / 1e3,
                    n_bolts=args.n_bolts, bolt_diameter=args.bolt_mm / 1e3,
                    safety_factor_yield=args.safety_factor,
                    safety_factor_ultimate=args.burst_safety_factor)


def describe(result: dict) -> str:
    if result['ok']:
        return f"structural OK (min margin {result['min_margin']:+.0%})"
    worst = ', '.join(f"{n} {result['checks'][n]['margin']:+.0%}" for n in result['failed'])
    return f"structural FAIL: {worst}"
