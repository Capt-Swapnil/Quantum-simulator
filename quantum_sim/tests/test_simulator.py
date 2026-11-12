"""Focused tests for the Simulator Wrapper component."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from quantum_sim.circuit_builder import build_superdense, build_teleportation_deferred
from quantum_sim.simulator import get_statevector, run_shots


TARGETS = {
    "|0>": [1, 0],
    "|1>": [0, 1],
    "|+>": [1 / np.sqrt(2), 1 / np.sqrt(2)],
    "|->": [1 / np.sqrt(2), -1 / np.sqrt(2)],
    "|+i>": [1 / np.sqrt(2), 1j / np.sqrt(2)],
    "|-i>": [1 / np.sqrt(2), -1j / np.sqrt(2)],
}


@pytest.mark.parametrize("psi", ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"])
def test_statevector_deferred_teleportation_preserves_input(psi):
    sv = get_statevector(build_teleportation_deferred(psi, upto_step=4))
    bob = partial_trace(sv, [0, 1])
    target = Statevector(TARGETS[psi])
    assert state_fidelity(bob, target) >= 0.9999


def test_statevector_accepts_arbitrary_normalized_input():
    psi = np.array([0.31 + 0.22j, -0.48 + 0.78j], dtype=complex)
    psi /= np.linalg.norm(psi)
    sv = get_statevector(build_teleportation_deferred(psi, upto_step=4))
    assert state_fidelity(partial_trace(sv, [0, 1]), Statevector(psi)) >= 0.9999


def test_statevector_rejects_measured_circuit_with_guidance():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="does not support measurements"):
        get_statevector(qc)


@pytest.mark.parametrize("message", ["00", "01", "10", "11"])
def test_run_shots_returns_expected_superdense_histogram(message):
    counts = run_shots(build_superdense(message, upto_step=4), shots=1024, seed=7)
    assert sum(counts.values()) == 1024
    assert len(counts) == 1
    assert next(iter(counts.values())) == 1024


@pytest.mark.parametrize("shots", [0, -1, 1.5, True])
def test_run_shots_rejects_invalid_shot_count(shots):
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="positive integer"):
        run_shots(qc, shots=shots)


def test_run_shots_requires_measurements():
    with pytest.raises(ValueError, match="requires a circuit with measurements"):
        run_shots(QuantumCircuit(1))
