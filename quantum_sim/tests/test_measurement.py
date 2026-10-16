import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from quantum_sim.measurement import (
    parse_bitstring, interpret, interpret_counts, correction_for, PROTOCOLS,
)


# ---- endianness (required by Dr. Maheswari) -------------------------------
def test_little_endian_rightmost_is_clbit0():
    assert parse_bitstring("01", ("c0", "c1")) == {"c0": 1, "c1": 0}
    assert parse_bitstring("10", ("c0", "c1")) == {"c0": 0, "c1": 1}
    assert parse_bitstring("100", ("a", "b", "c")) == {"a": 0, "b": 0, "c": 1}


def test_spaces_from_multiple_registers_are_ignored():
    # last-added register is leftmost: "1 0" -> reg2=1 (clbit1), reg1=0 (clbit0)
    assert parse_bitstring("1 0", ("c0", "c1")) == {"c0": 0, "c1": 1}


def test_endianness_matches_real_qiskit():
    """Flip ONLY qubit 0 / only qubit 1 and check against actual Aer output."""
    for flipped, expected_key in ((0, "01"), (1, "10")):
        qc = QuantumCircuit(2, 2)
        qc.x(flipped)
        qc.measure([0, 1], [0, 1])
        counts = AerSimulator().run(qc, shots=64).result().get_counts()
        assert list(counts) == [expected_key]
        bits = parse_bitstring(expected_key, ("c0", "c1"))
        assert bits[f"c{flipped}"] == 1 and bits[f"c{1 - flipped}"] == 0


def test_two_registers_real_qiskit():
    a, b = ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
    q = QuantumRegister(2)
    qc = QuantumCircuit(q, a, b)
    qc.x(0)                       # -> clbit 0 (register a)
    qc.measure(0, a[0]); qc.measure(1, b[0])
    key = next(iter(AerSimulator().run(qc, shots=8).result().get_counts()))
    assert " " in key
    assert parse_bitstring(key, ("a", "b")) == {"a": 1, "b": 0}


# ---- teleportation ---------------------------------------------------------
@pytest.mark.parametrize("bitstring,gates", [
    ("00", ()), ("01", ("Z",)), ("10", ("X",)), ("11", ("X", "Z")),
])
def test_teleportation_corrections(bitstring, gates):
    r = interpret("teleportation", bitstring)
    assert r.correction.gates == gates
    assert r.correction.target == "Bob"


def test_teleportation_named_bits():
    # "10": clbit1 (alice_m2)=1, clbit0 (alice_m1)=0
    r = interpret("teleportation", "10")
    assert r.bits == {"alice_m1": 0, "alice_m2": 1}
    assert r.correction.text == "Bob applies X"
    assert interpret("teleportation", "01").correction.text == "Bob applies Z"
    assert "no correction" in interpret("teleportation", "00").correction.text


# ---- superdense ------------------------------------------------------------
@pytest.mark.parametrize("bitstring,msg", [("00", "00"), ("01", "10"), ("10", "01"), ("11", "11")])
def test_superdense_decode_order(bitstring, msg):
    r = interpret("superdense", bitstring)
    assert r.message == msg and r.correction is None


# ---- swapping --------------------------------------------------------------
@pytest.mark.parametrize("bitstring,bell,gates", [
    ("00", "Φ⁺", ()), ("01", "Φ⁻", ("Z",)), ("10", "Ψ⁺", ("X",)), ("11", "Ψ⁻", ("X", "Z")),
])
def test_swapping(bitstring, bell, gates):
    r = interpret("swapping", bitstring)
    assert bell in r.summary and r.correction.gates == gates


# ---- counts / errors -------------------------------------------------------
def test_interpret_counts_sorted_and_normalised():
    rows = interpret_counts("teleportation", {"00": 200, "11": 600, "01": 224})
    assert [r["bitstring"] for r in rows] == ["11", "01", "00"]
    assert sum(r["probability"] for r in rows) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["0", "000", "0a", ""])
def test_bad_bitstrings(bad):
    with pytest.raises(ValueError):
        interpret("teleportation", bad)


def test_unknown_protocol():
    with pytest.raises(ValueError):
        interpret("bb84", "00")


def test_correction_for_identity_flag():
    assert correction_for(0, 0).is_identity and not correction_for(1, 0).is_identity
