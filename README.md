# Quantum Information Protocol Simulator

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Simulator Wrapper

The exact verification path accepts a measurement-free circuit and returns a
Qiskit `Statevector`. Use `build_teleportation_deferred` for teleportation
verification; it replaces Alice's classical feed-forward with coherent CX/CZ
corrections. The shot-based path accepts a measured circuit and returns Aer
counts. Use `quantum_sim.measurement.interpret` to convert Qiskit bitstrings to
named protocol outcomes.

```python
from quantum_sim.circuit_builder import build_teleportation_deferred, build_superdense
from quantum_sim.simulator import get_statevector, run_shots

state = get_statevector(build_teleportation_deferred("|+i>", upto_step=4))
counts = run_shots(build_superdense("11", upto_step=4), shots=1024)
```

Run all project tests with:

```bash
python -m pytest -q
```
