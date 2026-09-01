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


# KNSU propellant properties, sourced from the mech team's mini-project.
# Swap this out via a `propellant_props` argument if testing other grains.
DEFAULT_PROPELLANT = {
    'name': 'KNSU',
    'density': 1890,
    'tabs': [
        {'minPressure': 0, 'maxPressure': 7e7, 'a': 0.000101, 'n': 0.319,
         't': 1720, 'm': 41.98, 'k': 1.133}
    ]
}

# The GUI's actual default alert thresholds, taken from openMotor's
# uilib/defaults.py (DEFAULT_PREFERENCES['general']). motorlib.Motor.config
# properties default their `.value` to their *minimum* allowed value (0 for
# all three of these), NOT to any sensible real-world default - so leaving
# config unset makes every non-zero simulation trip all three alerts. This
# bit us on the first two sweep runs (every single row had all 4 alerts).
DEFAULT_MOTOR_CONFIG = {
    'maxPressure': 1500 * 6895,     # 1500 psi -> Pa
    'maxMassFlux': 2 / 0.001422,    # matches GUI's internal unit conversion
    'maxMachNumber': 0.7,
    'minPortThroat': 2,
    'flowSeparationWarnPercent': 0.05,
}


def run_bates_sim(params: BatesParams, propellant_props: dict = None,
                   nozzle_efficiency: float = 0.9, div_angle: float = 15,
                   conv_angle: float = 45) -> SimSummary:
    """Builds a single-BATES-grain motor from `params` and runs a full
    headless simulation. Returns summary stats plus raw curves."""
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
    # Replace with their exact logged values to cross-check against ~115.04 Ns.
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
