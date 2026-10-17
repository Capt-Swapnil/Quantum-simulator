
from quantum_sim.circuit_builder import (
    build_teleportation,
    build_superdense,
    build_swapping,
)

print("TELEPORTATION — STEP 2")
qc = build_teleportation("|+>", 2)
print(qc.draw(output="text"))

print("\nSUPERDENSE CODING — MESSAGE 11")
qc = build_superdense("11", 2)
print(qc.draw(output="text"))

print("\nENTANGLEMENT SWAPPING — STEP 1")
qc = build_swapping(1)
print(qc.draw(output="text"))
