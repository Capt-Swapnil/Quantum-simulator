"""End-to-end checks of the Measurement Handler on real protocol circuits.

Self-contained (own minimal circuits) so they pass before Swapnil's builder
exists. Once it does, swap the local builders for his and keep the asserts.
For each Alice/Charlie outcome we project the statevector, build the Qiskit
bitstring the way Aer would print it, let the handler pick the correction,
apply it, and check fidelity.
"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace, state_fidelity
from qiskit.quantum_info import random_statevector
from qiskit_aer import AerSimulator

from quantum_sim.measurement import interpret

GATES = {"X": XGate(), "Z": ZGate()}
OUTCOMES = [(m1, m2) for m1 in (0, 1) for m2 in (0, 1)]


def project(sv, qubit, outcome):
    p = np.zeros((2, 2)); p[outcome, outcome] = 1
    out = sv.evolve(Operator(p), [qubit])
    return Statevector(out.data / np.linalg.norm(out.data))


def qiskit_key(m1, m2):
    """What Aer prints: clbit0 (m1) rightmost."""
    return f"{m2}{m1}"


def apply_correction(sv, protocol, m1, m2, target_qubit):
    r = interpret(protocol, qiskit_key(m1, m2))
    assert (r.bits[list(r.bits)[0]], r.bits[list(r.bits)[1]]) == (m1, m2)
    for g in r.correction.gates:
        sv = sv.evolve(GATES[g], [target_qubit])
    return sv


def teleport_state(psi):
    qc = QuantumCircuit(3)           # q0 alice_in, q1 alice_bell, q2 bob
    qc.initialize(psi.data, 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    return Statevector(qc)


BASIS = {
    "0": [1, 0], "1": [0, 1],
    "+": [1 / np.sqrt(2), 1 / np.sqrt(2)], "-": [1 / np.sqrt(2), -1 / np.sqrt(2)],
    "+i": [1 / np.sqrt(2), 1j / np.sqrt(2)], "-i": [1 / np.sqrt(2), -1j / np.sqrt(2)],
}
STATES = [Statevector(v) for v in BASIS.values()] + [random_statevector(2, seed=s) for s in range(10)]


@pytest.mark.parametrize("psi", STATES)
@pytest.mark.parametrize("m1,m2", OUTCOMES)
def test_teleportation_with_handler_corrections(psi, m1, m2):
    sv = teleport_state(psi)
    sv = project(project(sv, 0, m1), 1, m2)
    sv = apply_correction(sv, "teleportation", m1, m2, 2)
    bob = partial_trace(DensityMatrix(sv), [0, 1])
    assert state_fidelity(bob, psi) >= 0.9999


def test_wrong_endianness_would_be_caught():
    """Swapping the two bits must break teleportation for at least one outcome."""
    psi = random_statevector(2, seed=7)  # generic state (a Y-eigenstate would hide the bug)
    broken = 0
    for m1, m2 in OUTCOMES:
        sv = project(project(teleport_state(psi), 0, m1), 1, m2)
        r = interpret("teleportation", f"{m1}{m2}")   # deliberately wrong key
        for g in r.correction.gates:
            sv = sv.evolve(GATES[g], [2])
        bob = partial_trace(DensityMatrix(sv), [0, 1])
        broken += state_fidelity(bob, psi) < 0.9999
    assert broken > 0


@pytest.mark.parametrize("m1,m2", OUTCOMES)
def test_swapping_with_handler_corrections(m1, m2):
    qc = QuantumCircuit(4)           # q0 alice, q1 charlie_1, q2 charlie_2, q3 bob
    qc.h(0); qc.cx(0, 1)
    qc.h(2); qc.cx(2, 3)
    qc.cx(1, 2); qc.h(1)             # Charlie's Bell measurement basis change
    sv = project(project(Statevector(qc), 1, m1), 2, m2)
    sv = apply_correction(sv, "swapping", m1, m2, 3)
    ab = partial_trace(DensityMatrix(sv), [1, 2])
    phi_plus = Statevector([1, 0, 0, 1] / np.sqrt(2))
    assert state_fidelity(ab, phi_plus) >= 0.9999
    # Charlie's qubits are a product state with Alice-Bob
    full = DensityMatrix(sv)
    c = partial_trace(full, [0, 3])
    assert np.isclose(np.real(np.trace(c.data @ c.data)), 1.0)  # pure => unentangled


@pytest.mark.parametrize("message", ["00", "01", "10", "11"])
def test_superdense_decoded_1024_of_1024(message):
    """Convention: message = 'xz' -> Alice applies X^x then Z^z. Bob measures
    qubit1 -> clbit0 (bob_m1 = x) and qubit0 -> clbit1 (bob_m2 = z)."""
    x, z = int(message[0]), int(message[1])
    qc = QuantumCircuit(2, 2)        # q0 alice, q1 bob
    qc.h(0); qc.cx(0, 1)
    if x: qc.x(0)
    if z: qc.z(0)
    qc.cx(0, 1); qc.h(0)
    qc.measure(1, 0); qc.measure(0, 1)
    counts = AerSimulator().run(qc, shots=1024).result().get_counts()
    assert len(counts) == 1
    assert interpret("superdense", next(iter(counts))).message == message
    assert next(iter(counts.values())) == 1024
