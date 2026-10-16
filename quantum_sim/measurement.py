"""Measurement Handler: Qiskit bitstrings -> named results -> corrections.

ENDIANNESS (the #1 source of bugs in these protocols)
-----------------------------------------------------
Qiskit prints classical bits with clbit 0 on the RIGHT:

    clbit index:   2  1  0
    bitstring:    "1  0  1"   -> clbit0=1, clbit1=0, clbit2=1

If a circuit has several ClassicalRegisters, counts keys contain spaces, and
the register added LAST appears LEFTMOST. Dropping the spaces and reversing
the string always gives bits in global clbit order (index 0 first), so we
name bits by their clbit index and do exactly that, in one place only
(`parse_bitstring`).

CONVENTION NEEDED FROM THE CIRCUIT BUILDER
------------------------------------------
Measure into clbits in the order listed in PROTOCOLS below, e.g. for
teleportation:  measure(alice_in -> clbit 0),  measure(alice_bell -> clbit 1).
"""
from __future__ import annotations

from dataclasses import dataclass

# Names listed in clbit-index order (index 0 first).
PROTOCOLS: dict[str, tuple[str, ...]] = {
    "teleportation": ("alice_m1", "alice_m2"),    # alice_in, alice_bell
    "superdense": ("bob_m1", "bob_m2"),           # see decode_superdense
    "swapping": ("charlie_m1", "charlie_m2"),     # charlie_1, charlie_2
}

# Bell-basis label of a (m1, m2) outcome. m1 = phase bit (Z), m2 = parity bit (X).
BELL_LABELS = {(0, 0): "Φ⁺", (1, 0): "Φ⁻", (0, 1): "Ψ⁺", (1, 1): "Ψ⁻"}


@dataclass(frozen=True)
class Correction:
    gates: tuple[str, ...]   # in the order they must be applied, e.g. ("X", "Z")
    target: str              # which party applies it
    text: str                # UI-ready sentence

    @property
    def is_identity(self) -> bool:
        return not self.gates


@dataclass(frozen=True)
class MeasurementResult:
    protocol: str
    bits: dict               # {"alice_m1": 0, "alice_m2": 1}
    correction: Correction | None
    summary: str
    message: str | None = None   # superdense only


# ---------------------------------------------------------------- parsing
def parse_bitstring(bitstring: str, names: tuple[str, ...] | list[str]) -> dict:
    """'10' with names ('a','b') -> {'a': 0, 'b': 1}  (clbit0 is rightmost)."""
    clean = bitstring.replace(" ", "")
    if len(clean) != len(names):
        raise ValueError(
            f"Bitstring {bitstring!r} has {len(clean)} bits but {len(names)} names given"
        )
    if set(clean) - {"0", "1"}:
        raise ValueError(f"Bitstring {bitstring!r} contains non-binary characters")
    return {name: int(bit) for name, bit in zip(names, reversed(clean))}


def _names(protocol: str) -> tuple[str, ...]:
    try:
        return PROTOCOLS[protocol]
    except KeyError:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {list(PROTOCOLS)}")


# ------------------------------------------------------------ corrections
def correction_for(m1: int, m2: int, target: str = "Bob") -> Correction:
    """Teleportation-style correction: X if m2 == 1, then Z if m1 == 1."""
    gates = tuple(g for g, flag in (("X", m2), ("Z", m1)) if flag)
    if not gates:
        return Correction((), target, f"{target} applies no correction (I)")
    return Correction(gates, target, f"{target} applies " + " then ".join(gates))


def decode_superdense(m1: int, m2: int) -> str:
    """Message string, read in the order m1 m2 (reverse of Qiskit's printed string)."""
    return f"{m1}{m2}"


# ------------------------------------------------------------- main API
def interpret(protocol: str, bitstring: str) -> MeasurementResult:
    bits = parse_bitstring(bitstring, _names(protocol))
    a, b = (bits[n] for n in PROTOCOLS[protocol])

    if protocol == "teleportation":
        c = correction_for(a, b)
        return MeasurementResult(protocol, bits, c, f"Alice measured ({a},{b}); {c.text}")

    if protocol == "swapping":
        c = correction_for(a, b)
        bell = BELL_LABELS[(a, b)]
        return MeasurementResult(
            protocol, bits, c,
            f"Charlie's Bell measurement gave |{bell}⟩; {c.text}",
        )

    # superdense: no correction, Bob just decodes
    msg = decode_superdense(a, b)
    return MeasurementResult(protocol, bits, None, f"Bob decoded message {msg}", message=msg)


def interpret_counts(protocol: str, counts: dict[str, int]) -> list[dict]:
    """Aer counts -> list of {result, count, probability}, most frequent first."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Empty counts")
    rows = [
        {"bitstring": k, "result": interpret(protocol, k), "count": v, "probability": v / total}
        for k, v in counts.items()
    ]
    return sorted(rows, key=lambda r: (-r["count"], r["bitstring"]))
