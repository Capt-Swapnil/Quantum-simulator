"""Structural tests for the QIPS circuit builder (requires qiskit + numpy)."""
import pytest
from quantum_sim.circuit_builder import (
    build_teleportation, build_teleportation_deferred,
    build_superdense, build_swapping,
)


def test_teleportation_steps_grow_monotonically():
    circuits = [build_teleportation("|+>", n) for n in range(1, 5)]
    assert all(c.num_qubits == 3 for c in circuits)
    assert [len(c.data) for c in circuits] == sorted(len(c.data) for c in circuits)
    assert circuits[0].qregs[0].name == "alice_in"


def test_deferred_teleportation_has_no_measurements():
    qc = build_teleportation_deferred("|+i>", 4)
    assert not any(item.operation.name == "measure" for item in qc.data)
    assert any(item.operation.name == "cz" for item in qc.data)


@pytest.mark.parametrize("message", ["00", "01", "10", "11"])
def test_superdense_valid_messages(message):
    qc = build_superdense(message, 4)
    assert qc.num_qubits == 2
    assert sum(item.operation.name == "measure" for item in qc.data) == 2


def test_superdense_rejects_invalid_message():
    with pytest.raises(ValueError):
        build_superdense("12", 2)


def test_swapping_register_roles():
    qc = build_swapping(2)
    names = [reg.name for reg in qc.qregs]
    assert names == ["alice", "charlie_1", "charlie_2", "bob"]


@pytest.mark.parametrize("step", [0, 5])
def test_invalid_steps_rejected(step):
    with pytest.raises(ValueError):
        build_teleportation("|0>", step)
