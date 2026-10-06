"""
Thin, headless wrapper around openMotor's motorlib for a single Star-grain
motor (a star-shaped core with `numPoints` triangular points). Same
params-in/curve-dict-out contract as bates_motor.py and finocyl_motor.py, so
the Design Agent's meta-search layer can call any of the wrappers
interchangeably and compare results across grain types.

Star-specific geometry, as motorlib's 'Star Grain' defines it: there is no
separate core circle. Each point is a triangle whose base (width
`pointWidth`) sits on the grain axis and whose tip reaches `pointLength` out
from the axis; the union of the points is the initial port. A star's burning
perimeter stays roughly constant while the points round off, which gives a
flatter (more neutral) thrust curve than a BATES grain of similar size, until
the flame front reaches the case and the slivers between the points burn out.
"""
from typing import TypedDict, List

import motorlib.motor
import motorlib.grains
import motorlib.propellant

from simulator_bridge.common import DEFAULT_PROPELLANT, DEFAULT_MOTOR_CONFIG


class StarParams(TypedDict):
    diameter: float      # grain OD, m
    length: float        # grain length, m
    numPoints: int       # number of star points
    pointLength: float   # radial reach of each point tip from the axis, m
    pointWidth: float    # width of each point's base at the axis, m
    throat: float        # nozzle throat diameter, m
    exit: float          # nozzle exit diameter, m


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


def run_star_sim(params: StarParams, propellant_props: dict = None,
                 nozzle_efficiency: float = 0.94,
                 div_angle: float = 15, conv_angle: float = 30,
                 throat_length: float = 0.0, erosion_coeff: float = 0.0,
                 slag_coeff: float = 0.0) -> SimSummary:
    """Builds a single-Star-grain motor from `params` and runs a full
    headless simulation. Returns summary stats plus raw curves - same shape
    as run_bates_sim() / run_finocyl_sim().

    throat_length, erosion_coeff, and slag_coeff default to 0 (idealized,
    non-eroding, zero-length throat) - see the matching note in
    bates_motor.run_bates_sim() for why these are explicit rather than left
    as silent motorlib defaults.
    """
    motor = motorlib.motor.Motor()

    grain = motorlib.grains.grainTypes['Star Grain']()
    grain.setProperties({
        'diameter': params['diameter'],
        'length': params['length'],
        'numPoints': int(round(params['numPoints'])),
        'pointLength': params['pointLength'],
        'pointWidth': params['pointWidth'],
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

    # Same fix as bates_motor.py: without this, config thresholds default to
    # 0 and every simulation trips every alert regardless of geometry.
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
    # Smoke-test geometry: same overall envelope as the BATES demo in
    # bates_motor.py, with a 6-point star core.
    demo_params: StarParams = {
        'diameter': 0.083058,
        'length': 0.1397,
        'numPoints': 6,
        'pointLength': 0.022,
        'pointWidth': 0.012,
        'throat': 0.01428,
        'exit': 0.045,
    }
    summary = run_star_sim(demo_params)
    print(f"Burn time: {summary['burn_time']:.3f} s")
    print(f"Peak thrust: {summary['peak_thrust']:.2f} N")
    print(f"Total impulse: {summary['total_impulse']:.2f} Ns")
    print(f"Peak Pc: {summary['peak_pressure']:.0f} Pa")
    print(f"Alerts: {summary['alerts']}")
