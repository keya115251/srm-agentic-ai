"""
Thin, headless wrapper around openMotor's motorlib for a single BATES-grain
motor. This is the function the Design Agent's tool layer and the surrogate
model's data generator both call: params in, curve dict out.

Requires openMotor's motorlib to be importable (either vendored into this
repo, or installed/available on PYTHONPATH). motorlib has no Qt dependency,
so this runs fully headless - see docs/openmotor_scriptability_validation.md.
"""
from typing import TypedDict, List

import motorlib.motor
import motorlib.grains
import motorlib.propellant

from simulator_bridge.common import DEFAULT_PROPELLANT, DEFAULT_MOTOR_CONFIG


class BatesParams(TypedDict):
    diameter: float        # grain OD, m
    length: float           # grain length, m
    coreDiameter: float     # core diameter, m
    throat: float            # nozzle throat diameter, m
    exit: float               # nozzle exit diameter, m


class SimSummary(TypedDict):
    success: bool
    alerts: List[str]
    burn_time: float
    peak_thrust: float
    total_impulse: float
    peak_pressure: float
    peak_kn: float
    time: List[float]
    thrust: List[float]
    pressure: List[float]
    kn: List[float]


def run_bates_sim(params: BatesParams, propellant_props: dict = None,
                   nozzle_efficiency: float = 0.94, div_angle: float = 15,
                   conv_angle: float = 30, throat_length: float = 0.0,
                   erosion_coeff: float = 0.0, slag_coeff: float = 0.0) -> SimSummary:
    """Builds a single-BATES-grain motor from `params` and runs a full
    headless simulation. Returns summary stats plus raw curves.

    throat_length, erosion_coeff, and slag_coeff default to 0, which models
    an idealized, perfectly rigid, non-eroding throat with zero length. This
    matches motorlib's own FloatProperty minimums (i.e. what happens if you
    never touch these fields in the GUI either), so it's a deliberate,
    documented simplification rather than an unexamined default - not a
    physically realistic throat. Pass real values here once available (e.g.
    from the mech team's nozzle material/geometry) for a more accurate sim,
    particularly for burn-time-integrated effects like throat erosion, which
    Puskulcu & Ulas (2008) identified as a real source of sim-vs-test-fire
    pressure deviation.
    """
    motor = motorlib.motor.Motor()

    grain = motorlib.grains.BatesGrain()
    grain.setProperties({
        'diameter': params['diameter'],
        'length': params['length'],
        'coreDiameter': params['coreDiameter'],
        'inhibitedEnds': 'Neither',
    })
    motor.grains.append(grain)

    motor.nozzle.setProperties({
        'throat': params['throat'],
        'exit': params['exit'],
        'efficiency': nozzle_efficiency,
        'divAngle': div_angle,
        'convAngle': conv_angle,
        'throatLength': throat_length,
        'erosionCoeff': erosion_coeff,
        'slagCoeff': slag_coeff,
    })

    motor.propellant = motorlib.propellant.Propellant()
    motor.propellant.setProperties(propellant_props or DEFAULT_PROPELLANT)

    # Without this, config thresholds default to 0 (their FloatProperty
    # minimum) and every simulation trips every alert regardless of geometry.
    motor.config.setProperties(DEFAULT_MOTOR_CONFIG)

    result = motor.runSimulation()
    channels = result.channels

    time = list(channels['time'].getData())
    thrust = list(channels['force'].getData())
    pressure = list(channels['pressure'].getData())
    kn = list(channels['kn'].getData())

    impulse = sum((thrust[i] + thrust[i - 1]) / 2 * (time[i] - time[i - 1])
                   for i in range(1, len(time)))

    return SimSummary(
        success=bool(getattr(result, 'success', True)),
        alerts=[a.description for a in getattr(result, 'alerts', [])],
        burn_time=time[-1] if time else 0.0,
        peak_thrust=max(thrust) if thrust else 0.0,
        total_impulse=impulse,
        peak_pressure=max(pressure) if pressure else 0.0,
        peak_kn=max(kn) if kn else 0.0,
        time=time, thrust=thrust, pressure=pressure, kn=kn,
    )


if __name__ == '__main__':
    # Quick smoke test using the mech team's approximate G-class parameters.
    demo_params: BatesParams = {
        'diameter': 0.083058,
        'length': 0.1397,
        'coreDiameter': 0.05,
        'throat': 0.01428,
        'exit': 0.045,
    }
    summary = run_bates_sim(demo_params)
    print(f"Burn time: {summary['burn_time']:.3f} s")
    print(f"Peak thrust: {summary['peak_thrust']:.2f} N")
    print(f"Total impulse: {summary['total_impulse']:.2f} Ns")
    print(f"Peak Pc: {summary['peak_pressure']:.0f} Pa")
    print(f"Alerts: {summary['alerts']}")

    