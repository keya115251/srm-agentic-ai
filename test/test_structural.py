"""
Verification of structural_tools/casing.py against the standard results it
claims to implement (force equilibrium, thin-wall limit, Roark plate
coefficients) plus a hand-worked example. Run with: python -m pytest test/
"""
import math
import os
import sys

import pytest
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structural_tools.casing import (CaseSpec, barlow_burst_pressure, check_structure,
                                     flat_plate_max_stress, lame_stresses, thin_wall_stresses,
                                     von_mises)

P = 6e6
R_I = 0.05


@pytest.mark.parametrize('t', [0.001, 0.005, 0.02, 0.05])
def test_lame_hoop_satisfies_half_cylinder_equilibrium(t):
    # Cutting the cylinder in half: the hoop stress integrated through the
    # wall must carry the pressure force on the bore, p * r_i per unit length.
    r_o = R_I + t
    force, _ = quad(lambda r: lame_stresses(P, R_I, r_o, r)['hoop'], R_I, r_o)
    assert force == pytest.approx(P * R_I, rel=1e-9)


def test_lame_axial_satisfies_end_cap_equilibrium():
    r_o = R_I + 0.01
    axial = lame_stresses(P, R_I, r_o)['axial']
    assert axial * math.pi * (r_o ** 2 - R_I ** 2) == pytest.approx(P * math.pi * R_I ** 2)


def test_lame_boundary_conditions():
    r_o = R_I + 0.01
    assert lame_stresses(P, R_I, r_o, R_I)['radial'] == pytest.approx(-P)
    assert lame_stresses(P, R_I, r_o, r_o)['radial'] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize('d_over_t,max_err', [(20, 0.06), (100, 0.011), (1000, 0.0011)])
def test_lame_converges_to_thin_wall(d_over_t, max_err):
    t = 2 * R_I / d_over_t
    thin = thin_wall_stresses(P, R_I, t)['hoop']
    thick = lame_stresses(P, R_I, R_I + t)['hoop']
    assert thick > thin
    assert (thick - thin) / thin < max_err


def test_von_mises_uniaxial_and_pure_shear():
    assert von_mises(100, 0, 0) == pytest.approx(100)
    assert von_mises(50, -50, 0) == pytest.approx(50 * math.sqrt(3))


def test_flat_plate_roark_coefficients():
    # Roark Table 11.2: sigma = beta * q a^2 / t^2 with beta = 0.75 (clamped)
    # and 1.2375 (simply supported, nu = 0.3).
    q, a, t = 1e6, 0.05, 0.01
    assert flat_plate_max_stress(q, a, t, 0.3, 'clamped') == pytest.approx(0.75 * q * a ** 2 / t ** 2)
    assert flat_plate_max_stress(q, a, t, 0.3) == pytest.approx(1.2375 * q * a ** 2 / t ** 2)


def test_barlow():
    assert barlow_burst_pressure(0.003, 0.1, 300e6) == pytest.approx(18e6)


def test_worked_example():
    # 98 mm grain in the default case (3.175 mm 6061-T6 wall, 12.7 mm
    # simply supported bulkhead, 12 x M6 bolts) at 6 MPa, by hand:
    #   r_i = 0.049, r_o = 0.052175
    #   bore hoop = 6e6 (r_o^2 + r_i^2) / (r_o^2 - r_i^2) = 95.7 MPa
    #   bulkhead  = 3 (3.33) 6e6 0.049^2 / (8 0.0127^2) = 111.6 MPa
    #   bolt load = 6e6 pi 0.049^2 = 45.26 kN, shear = 45.26e3 / (12 pi 0.003^2) = 133.4 MPa
    #   bearing   = 45.26e3 / (12 0.006 0.003175) = 198.0 MPa
    r = check_structure({'diameter': 0.098}, 6e6)
    c = r['checks']
    assert lame_stresses(6e6, 0.049, 0.052175)['hoop'] == pytest.approx(95.7e6, rel=2e-3)
    assert c['bulkhead_bending']['stress'] == pytest.approx(111.6e6, rel=2e-3)
    assert c['bolt_shear']['stress'] == pytest.approx(133.4e6, rel=2e-3)
    assert c['bolt_bearing']['stress'] == pytest.approx(198.0e6, rel=2e-3)
    # allowable shear 640 / sqrt(3) / 1.5 = 246 MPa, so bolts pass with ~85% margin
    assert c['bolt_shear']['margin'] == pytest.approx(246.3 / 133.4 - 1, abs=0.01)
    assert r['ok']


def test_flags_overpressure_and_thin_wall():
    assert not check_structure({'diameter': 0.098}, 20e6)['ok']
    thin = check_structure({'diameter': 0.098}, 6e6, CaseSpec(wall_thickness=0.5e-3))
    assert 'case_von_mises' in thin['failed']
