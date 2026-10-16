# Quantum Simulator
## Quantum Information Protocol Simulator

# Measurement Handler (`quantum_sim/measurement.py`)

Component D of Phase 1, owned by Samruddhi.

Turns Qiskit measurement bitstrings into named results and the correction each party must apply, for all three protocols (teleportation, superdense coding, entanglement swapping).

## Why this exists

Qiskit bitstrings are **little-endian**: classical bit 0 is the rightmost character. Getting this wrong silently breaks the corrections. All bit-order handling lives in one function, `parse_bitstring`, so the rest of the project never has to think about it.

```
clbit index:   2  1  0
bitstring:    "1  0  1"   ->  clbit0 = 1, clbit1 = 0, clbit2 = 1
```

If a circuit uses several `ClassicalRegister`s, Aer puts spaces between them and the register added last appears leftmost. The handler strips the spaces and reverses the string, which always gives global clbit order.

## API

| Function | What it does |
| --- | --- |
| `parse_bitstring(bitstring, names)` | `"10"`, `("a","b")` -> `{"a": 0, "b": 1}` |
| `interpret(protocol, bitstring)` | Returns a `MeasurementResult`: named `bits`, `correction`, `summary`, and `message` (superdense only) |
| `interpret_counts(protocol, counts)` | Aer histogram -> list of `{bitstring, result, count, probability}`, most frequent first |
| `correction_for(m1, m2, target="Bob")` | Teleportation-style correction as a `Correction` |
| `decode_superdense(m1, m2)` | Message string, e.g. `"10"` |

`protocol` is one of `"teleportation"`, `"superdense"`, `"swapping"`. Bad bitstrings (wrong length, non-binary characters) or an unknown protocol raise `ValueError`.

## Conventions the circuits must follow

These are the contract between the Circuit Builder and this module. Bits are listed in **clbit index order** (index 0 first).

| Protocol | clbit 0 | clbit 1 |
| --- | --- | --- |
| Teleportation | `alice_m1` (measures `alice_in`) | `alice_m2` (measures `alice_bell`) |
| Superdense | `bob_m1` | `bob_m2` |
| Swapping | `charlie_m1` (measures `charlie_1`) | `charlie_m2` (measures `charlie_2`) |

### Corrections

Bob (the far qubit) applies **X if m2 = 1, then Z if m1 = 1**.

| m1 | m2 | Correction |
| --- | --- | --- |
| 0 | 0 | none (I) |
| 1 | 0 | Z |
| 0 | 1 | X |
| 1 | 1 | X then Z |

For swapping, `(m1, m2)` also names Charlie's Bell outcome: (0,0) Φ⁺, (1,0) Φ⁻, (0,1) Ψ⁺, (1,1) Ψ⁻. This assumes Charlie does CX(charlie_1, charlie_2) then H(charlie_1) before measuring.

### Superdense coding

Message `xz` means Alice applies X^x then Z^z to her qubit. Bob undoes the Bell pair with CX(alice, bob) then H(alice), and measures **bob's qubit into clbit 0** and **Alice's qubit into clbit 1**. The decoded message is `f"{m1}{m2}"`, which is the **reverse** of the printed Qiskit string: the bitstring `"01"` decodes to message `10`. Message 11 is encoded as ZX; Y differs only by a global phase.

## Usage

```python
from quantum_sim.measurement import interpret, interpret_counts

r = interpret("teleportation", "10")
r.bits                  # {'alice_m1': 0, 'alice_m2': 1}
r.correction.gates      # ('X',)
r.correction.text       # 'Bob applies X'

interpret("swapping", "11").summary
# "Charlie's Bell measurement gave |Ψ⁻⟩; Bob applies X then Z"

interpret("superdense", "01").message   # '10'

# From a shot-based run
counts = {"00": 260, "01": 250, "10": 270, "11": 244}
for row in interpret_counts("teleportation", counts):
    print(row["bitstring"], row["result"].correction.text, row["probability"])
```

## Tests

```bash
pip install qiskit qiskit-aer pytest
python -m pytest quantum_sim/tests -q
```

- `tests/test_measurement.py`: unit tests. Little-endian mapping (including against real Aer output and the two-register case), every outcome of each protocol, bad input, and counts ordering.
- `tests/test_measurement_integration.py`: end-to-end on real protocol circuits.
  - Teleportation: 16 states x 4 outcomes, fidelity >= 0.9999 after the handler's correction.
  - Swapping: all 4 outcomes give Alice-Bob |Φ⁺⟩ fidelity >= 0.9999, with Charlie's qubits unentangled.
  - Superdense: all 4 messages decode correctly in 1024/1024 shots.
  - Endianness guard: a deliberately flipped bit order must fail teleportation.
