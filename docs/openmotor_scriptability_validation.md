# OpenMotor Scriptability Validation (Week 1 risk item)

**Status: PASSED.** OpenMotor's simulation core (`motorlib`) runs fully headless -
no PyQt, no Qt, no GUI process required.

## What was checked

1. `motorlib`'s source has zero imports of PyQt5/PyQt6 anywhere. All its GUI-free
   dependencies are `numpy`, `scipy`, `scikit-fmm`, `scikit-image`.
2. The one native-code dependency, `mathlib._find_perimeter_cy`, is a plain Cython
   extension (perimeter-finding for FMM grains) with no Qt/GUI ties. Builds cleanly
   via `python setup.py build_ext --inplace`.
3. `import motorlib` in a bare Python shell loads clean; confirmed via
   `'PyQt6' not in sys.modules` after import.
4. Built a full motor object in code (`Motor()`, `BatesGrain()`, `Propellant()`,
   `Nozzle`) with `.setProperties()` dicts, called `motor.runSimulation()`, and
   read back thrust/pressure/Kn/time curves from the returned result's
   `.channels` dict. No file I/O, no GUI, no subprocess.

## How to reproduce

```bash
git clone https://github.com/reilleya/openMotor
cd openMotor
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-fmm scikit-image cython
python setup.py build_ext --inplace   # builds mathlib's Cython extension
python3 scripts/validate_headless.py  # or simulator_bridge/bates_motor.py
```

## What this means for the project

- The simulator bridge can call `motorlib` directly as a Python library -
  no subprocess/CLI wrapping, no screen-scraping GUI output.
- `simulator_bridge/bates_motor.py` is a first working wrapper: params in
  (grain diameter/length/core, nozzle throat/exit) -> curve dict out
  (time, thrust, pressure, Kn + summary stats).
- This directly unblocks Phase 2 (batch data generation for the surrogate
  model) and the Design Agent's tool layer - both were gated on this.
- Fallback to reimplementing Nakka/Sutton equations from scratch is **not
  needed**.

## Not yet done

- Cross-check against the mech team's exact G109 (~115.04 Ns) and G104
  (~110.01 Ns) parameters to confirm numeric agreement with their GUI-run
  results, not just "the simulation runs."
- Vendor `motorlib`/`mathlib` properly into this repo (as a git submodule or
  pinned copy) rather than the ad-hoc copy used for this validation.
- Test Finocyl and other non-BATES grain types the mech team may also use.
- Check licensing implications (GPLv3) for vendoring into this project's repo.
