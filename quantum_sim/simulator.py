"""Simulation helpers for exact verification and shot-based protocol demos.

``get_statevector`` is the exact, measurement-free verification path. Use the
deferred-measurement circuits from :mod:`quantum_sim.circuit_builder` for
teleportation verification. ``run_shots`` is the Aer-based display path for
circuits that contain measurements.
"""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


def _require_circuit(qc: QuantumCircuit) -> None:
    if not isinstance(qc, QuantumCircuit):
        raise TypeError("qc must be a qiskit.QuantumCircuit")


def get_statevector(qc: QuantumCircuit) -> Statevector:
    """Return the exact final state of a unitary circuit.

    Circuits containing measurements are rejected because a statevector cannot
    represent the post-measurement classical mixture. For teleportation,
    ``build_teleportation_deferred`` provides the coherent verification form.
    """
    _require_circuit(qc)
    if any(instruction.operation.name == "measure" for instruction in qc.data):
        raise ValueError(
            "Statevector simulation does not support measurements; use a "
            "measurement-free/deferred-measurement circuit"
        )
    try:
        return Statevector.from_instruction(qc)
    except Exception as exc:
        raise ValueError(
            "Statevector simulation requires a unitary circuit without "
            "measurement or classical control"
        ) from exc


def run_shots(
    qc: QuantumCircuit,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, int]:
    """Run a measured circuit on Aer and return its bitstring counts.

    ``seed`` is optional and makes sampling reproducible when supplied.
    Bitstring ordering is left as emitted by Qiskit; use
    :func:`quantum_sim.measurement.interpret` to map it to named bits.
    """
    _require_circuit(qc)
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if not any(instruction.operation.name == "measure" for instruction in qc.data):
        raise ValueError("run_shots requires a circuit with measurements")

    # Keep Aer's native runtime out of the exact-state path. This also makes
    # circuit construction/statevector use independent of the Aer install.
    from qiskit_aer import AerSimulator

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    return result.get_counts(compiled)
