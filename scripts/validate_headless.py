"""
Validates that openMotor's simulation core (motorlib) can be driven
entirely headlessly - no GUI, no Qt - which is the Week-1 risk item
for the Design Agent's simulator bridge.

Builds a BATES motor matching the mech team's known G-class KNSB config
and runs a full simulation, printing summary stats (total impulse,
peak thrust, peak Pc, burn time) plus a few rows of the raw curves.
"""
import sys
import motorlib.motor
import motorlib.grains
import motorlib.propellant

assert 'PyQt6' not in sys.modules and 'PyQt5' not in sys.modules, "Qt should never load in this path"

# --- Build the motor ---
motor = motorlib.motor.Motor()

# BATES grain - replace these with the mech team's exact G109/G104 values
# to cross-check against their known OpenMotor GUI results (~115.04 Ns / ~110.01 Ns)
grain = motorlib.grains.BatesGrain()
grain.setProperties({
    'diameter': 0.083058,      # m
    'length': 0.1397,          # m
    'coreDiameter': 0.05,      # m
    'inhibitedEnds': 'Neither'
})
motor.grains.append(grain)

motor.nozzle.setProperties({
    'throat': 0.01428,   # m
    'exit': 0.045,        # m
    'efficiency': 0.9,
    'divAngle': 15,
    'convAngle': 45,
})

motor.propellant = motorlib.propellant.Propellant()
motor.propellant.setProperties({
    'name': 'KNSU',
    'density': 1890,
    'tabs': [
        {'minPressure': 0, 'maxPressure': 7e7, 'a': 0.000101, 'n': 0.319, 't': 1720, 'm': 41.98, 'k': 1.133}
    ]
})

# --- Check for geometry/config errors before running ---
errors = motor.getGeometryErrors() if hasattr(motor, 'getGeometryErrors') else []
if errors:
    print("Geometry errors:", errors)

# --- Run the simulation ---
result = motor.runSimulation()

print("=== Headless simulation result ===")
print("Success:", result.success if hasattr(result, 'success') else 'n/a')
print("Alerts:", [a.description for a in result.alerts] if hasattr(result, 'alerts') else 'n/a')

channels = result.channels
print("\nAvailable output channels:", list(channels.keys()))

time = channels['time'].getData()
thrust = channels['force'].getData() if 'force' in channels else None
pressure = channels['pressure'].getData() if 'pressure' in channels else None
kn = channels['kn'].getData() if 'kn' in channels else None

print(f"\nBurn time: {time[-1]:.3f} s")
if thrust is not None:
    print(f"Peak thrust: {max(thrust):.2f} N")
    impulse = sum((thrust[i] + thrust[i-1]) / 2 * (time[i] - time[i-1]) for i in range(1, len(time)))
    print(f"Total impulse (trapezoidal): {impulse:.2f} Ns")
if pressure is not None:
    print(f"Peak chamber pressure: {max(pressure):.0f} Pa")
if kn is not None:
    print(f"Peak Kn: {max(kn):.1f}")

print("\nFirst 5 timesteps (time, thrust, pressure):")
for i in range(min(5, len(time))):
    t_ = thrust[i] if thrust is not None else float('nan')
    p_ = pressure[i] if pressure is not None else float('nan')
    print(f"  t={time[i]:.4f}s  thrust={t_:.2f}N  Pc={p_:.0f}Pa")
