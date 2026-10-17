"""Step-wise Qiskit circuit construction for QIPS protocols.

This module constructs circuits only; simulation, fidelity analysis, and
interpretation of measurement strings belong to other modules.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit


_TELEPORT_STEPS = 4
_SUPERDENSE_STEPS = 4
_SWAPPING_STEPS = 4


def _validate_step(upto_step: int, max_step: int) -> None:
    if not isinstance(upto_step, int) or isinstance(upto_step, bool):
        raise TypeError("upto_step must be an integer")
    if not 1 <= upto_step <= max_step:
        raise ValueError(f"upto_step must be between 1 and {max_step}")


def _prepare_state(qc: QuantumCircuit, qubit: Qubit, psi: str | Sequence[complex]) -> None:
    """Prepare a supported named state or a normalized two-amplitude state."""
    named = {
        "0": (1, 0), "|0>": (1, 0),
        "1": (0, 1), "|1>": (0, 1),
        "+": (1 / np.sqrt(2), 1 / np.sqrt(2)),
        "|+>": (1 / np.sqrt(2), 1 / np.sqrt(2)),
        "-": (1 / np.sqrt(2), -1 / np.sqrt(2)),
        "|->": (1 / np.sqrt(2), -1 / np.sqrt(2)),
        "+i": (1 / np.sqrt(2), 1j / np.sqrt(2)),
        "|+i>": (1 / np.sqrt(2), 1j / np.sqrt(2)),
        "-i": (1 / np.sqrt(2), -1j / np.sqrt(2)),
        "|-i>": (1 / np.sqrt(2), -1j / np.sqrt(2)),
    }
    if isinstance(psi, str):
        key = psi.strip().replace(" ", "")
        if key not in named:
            raise ValueError(f"Unsupported state label: {psi!r}")
        amplitudes = np.asarray(named[key], dtype=complex)
    else:
        amplitudes = np.asarray(psi, dtype=complex)
    if amplitudes.shape != (2,):
        raise ValueError("psi must be a supported state label or two amplitudes")
    norm = np.linalg.norm(amplitudes)
    if not np.isclose(norm, 1.0, atol=1e-9):
        raise ValueError("psi amplitudes must be normalized")
    # StatePreparation is avoided to keep the circuit's preparation transparent.
    alpha, beta = amplitudes / norm
    theta = 2 * np.arctan2(abs(beta), abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    qc.u(float(theta), float(phi), 0.0, qubit)


def build_teleportation(
    psi: str | Sequence[complex] = "|0>",
    upto_step: int = 1,
) -> QuantumCircuit:
    """Build measured teleportation circuit through step 1–4.

    Steps: 1 input-state preparation; 2 Bell-pair preparation;
    3 Alice's Bell-basis operations and measurements; 4 Bob's corrections.
    """
    _validate_step(upto_step, _TELEPORT_STEPS)
    alice_in = QuantumRegister(1, "alice_in")
    alice_bell = QuantumRegister(1, "alice_bell")
    bob = QuantumRegister(1, "bob")
    alice_m1 = ClassicalRegister(1, "alice_m1")
    alice_m2 = ClassicalRegister(1, "alice_m2")
    qc = QuantumCircuit(alice_in, alice_bell, bob, alice_m1, alice_m2,
                        name="teleportation")

    _prepare_state(qc, alice_in[0], psi)
    if upto_step >= 2:
        qc.h(alice_bell[0])
        qc.cx(alice_bell[0], bob[0])
    if upto_step >= 3:
        qc.cx(alice_in[0], alice_bell[0])
        qc.h(alice_in[0])
        qc.measure(alice_in[0], alice_m1[0])
        qc.measure(alice_bell[0], alice_m2[0])
    if upto_step >= 4:
        with qc.if_test((alice_m2[0], 1)):
            qc.x(bob[0])
        with qc.if_test((alice_m1[0], 1)):
            qc.z(bob[0])
    return qc


def build_teleportation_deferred(
    psi: str | Sequence[complex] = "|0>",
    upto_step: int = 3,
) -> QuantumCircuit:
    """Build a measurement-free teleportation circuit for statevector checks.

    At step 3, Alice's Bell-basis operations are retained but measurements
    are deferred. Step 4 coherently applies CX/CZ controlled by Alice's
    measurement qubits. This is a verification representation, not the
    shot-based demo circuit.
    """
    _validate_step(upto_step, _TELEPORT_STEPS)
    alice_in = QuantumRegister(1, "alice_in")
    alice_bell = QuantumRegister(1, "alice_bell")
    bob = QuantumRegister(1, "bob")
    qc = QuantumCircuit(alice_in, alice_bell, bob, name="teleportation_deferred")
    _prepare_state(qc, alice_in[0], psi)
    if upto_step >= 2:
        qc.h(alice_bell[0])
        qc.cx(alice_bell[0], bob[0])
    if upto_step >= 3:
        qc.cx(alice_in[0], alice_bell[0])
        qc.h(alice_in[0])
    if upto_step >= 4:
        qc.cx(alice_bell[0], bob[0])
        qc.cz(alice_in[0], bob[0])
    return qc


def build_superdense(message: str = "00", upto_step: int = 1) -> QuantumCircuit:
    """Build a step-wise superdense-coding circuit.

    Encoding convention: 00→I, 01→X, 10→Z, 11→ZX on Alice's qubit.
    """
    _validate_step(upto_step, _SUPERDENSE_STEPS)
    if message not in {"00", "01", "10", "11"}:
        raise ValueError("message must be one of: '00', '01', '10', '11'")
    alice = QuantumRegister(1, "alice")
    bob = QuantumRegister(1, "bob")
    decoded = ClassicalRegister(2, "decoded")
    qc = QuantumCircuit(alice, bob, decoded, name=f"superdense_{message}")

    if upto_step >= 1:
        qc.h(alice[0])
        qc.cx(alice[0], bob[0])
    if upto_step >= 2:
        if message == "01":
            qc.x(alice[0])
        elif message == "10":
            qc.z(alice[0])
        elif message == "11":
            qc.z(alice[0])
            qc.x(alice[0])
    if upto_step >= 3:
        # Bob applies the inverse Bell-pair preparation.
        qc.cx(alice[0], bob[0])
        qc.h(alice[0])
    if upto_step >= 4:
        # Qiskit displays classical bit 1 left of bit 0; map qubits explicitly.
        qc.measure(alice[0], decoded[1])
        qc.measure(bob[0], decoded[0])
    return qc


def build_swapping(upto_step: int = 1) -> QuantumCircuit:
    """Build a step-wise entanglement-swapping circuit.

    Steps: 1 create Alice–Charlie and Charlie–Bob Bell pairs;
    2 Charlie's Bell-basis operations and measurements;
    3 apply corrections to Bob; 4 retained as a named completion stage.
    """
    _validate_step(upto_step, _SWAPPING_STEPS)
    alice = QuantumRegister(1, "alice")
    charlie_1 = QuantumRegister(1, "charlie_1")
    charlie_2 = QuantumRegister(1, "charlie_2")
    bob = QuantumRegister(1, "bob")
    charlie_m1 = ClassicalRegister(1, "charlie_m1")
    charlie_m2 = ClassicalRegister(1, "charlie_m2")
    qc = QuantumCircuit(alice, charlie_1, charlie_2, bob,
                        charlie_m1, charlie_m2, name="entanglement_swapping")

    if upto_step >= 1:
        qc.h(alice[0])
        qc.cx(alice[0], charlie_1[0])
        qc.h(charlie_2[0])
        qc.cx(charlie_2[0], bob[0])
    if upto_step >= 2:
        qc.cx(charlie_1[0], charlie_2[0])
        qc.h(charlie_1[0])
        qc.measure(charlie_1[0], charlie_m1[0])
        qc.measure(charlie_2[0], charlie_m2[0])
    if upto_step >= 3:
        # Standard correction convention for this register/gate ordering.
        with qc.if_test((charlie_m2[0], 1)):
            qc.x(bob[0])
        with qc.if_test((charlie_m1[0], 1)):
            qc.z(bob[0])
    # Step 4 is reserved as the protocol's final/verification stage; no
    # additional gates belong here because verification is outside this module.
    return qc
