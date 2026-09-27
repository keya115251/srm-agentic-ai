"""
Shared constants used by every grain-type wrapper in simulator_bridge/
(bates_motor.py, finocyl_motor.py, and future ones). Centralized here so
propellant properties and safety-threshold defaults don't drift between
wrappers as more grain types get added.
"""

# KNSU propellant properties. Values a/n/k/t/m are Nakka's published KNSU
# constants; density was cross-checked against the mech team's actual
# OpenMotor library entry ("Nakka - KNSU": 0.065029 lb/in^3) and corrected
# from an earlier value of 1890 kg/m^3, which was wrong by ~5%.
# Swap this out via a `propellant_props` argument on any wrapper's run_*_sim()
# function if testing other propellants.
DEFAULT_PROPELLANT = {
    'name': 'KNSU',
    'density': 1800,
    'tabs': [
        {'minPressure': 0, 'maxPressure': 7e7, 'a': 0.000101, 'n': 0.319,
         't': 1720, 'm': 41.98, 'k': 1.133}
    ]
}

# The GUI's actual default config values, taken directly from openMotor's
# uilib/defaults.py (DEFAULT_PREFERENCES['general']). motorlib.Motor.config
# properties default their `.value` to their *minimum* allowed value unless
# explicitly set - which for most of these silently produces nonphysical
# behavior rather than an error:
#   - maxPressure/maxMassFlux/maxMachNumber/minPortThroat/
#     flowSeparationWarnPercent defaulting to 0 makes every simulation trip
#     every safety alert regardless of geometry (found first, see git history).
#   - ambPressure defaulting to ~0 (near-vacuum, its FloatProperty minimum of
#     0.0001 Pa) instead of sea-level (101325 Pa) silently inflates the
#     nozzle's pressure-thrust term, overstating thrust/impulse by ~15-20%
#     with NO warning or alert - found by cross-checking against the mech
#     team's real OpenMotor GUI runs, where Kn and chamber pressure matched
#     almost exactly but thrust/impulse did not.
# All keys below are included for completeness, even ones whose default
# happens to equal motorlib's own minimum, so nothing is silently assumed
# again.
DEFAULT_MOTOR_CONFIG = {
    'maxPressure': 1500 * 6895,       # 1500 psi -> Pa
    'maxMassFlux': 2 / 0.001422,      # matches GUI's internal unit conversion
    'maxMachNumber': 0.7,
    'minPortThroat': 2,
    'flowSeparationWarnPercent': 0.05,
    'burnoutWebThres': 0.001 / 39.37, # = motorlib's own minimum; included for completeness
    'burnoutThrustThres': 0.1,        # percent of peak thrust that counts as burnout
    'timestep': 0.03,                 # seconds
    'ambPressure': 101325,            # Pa (sea level) - see note above; this was the
                                       # silent bug behind our thrust/impulse mismatch
    'mapDim': 750,                    # grain-map grid resolution
    'sepPressureRatio': 0.4,          # Summerfield criteria flow-separation ratio
}