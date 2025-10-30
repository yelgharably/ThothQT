"""
Debug the H00 singularity issue.
"""
import numpy as np
from graphene_builder_v2 import GrapheneRibbonBuilder

builder = GrapheneRibbonBuilder(a=1.42, t=-2.7)
device, _ = builder.zigzag_ribbon(width=2, length=3)

H00 = device.left.H00
H01 = device.left.H01

print("=" * 80)
print("ZIGZAG RIBBON H00 ANALYSIS")
print("=" * 80)
print(f"\nH00 (4×4):\n{H00}")
print(f"\nH01 (4×4):\n{H01}")

# Check for duplicate rows
print("\nRow Analysis:")
for i in range(4):
    for j in range(i+1, 4):
        if np.allclose(H00[i, :], H00[j, :]):
            print(f"  Row {i} == Row {j}")

# Check rank
rank = np.linalg.matrix_rank(H00)
print(f"\nRank of H00: {rank}/4")
print(f"det(H00) = {np.linalg.det(H00):.3e}")

# Show connectivity
print("\nConnectivity Pattern:")
for i in range(4):
    neighbors = np.where(np.abs(H00[i, :]) > 0.1)[0]
    print(f"  Atom {i}: connects to {neighbors.tolist()}")

# Compare with KWANT
print("\n" + "=" * 80)
print("KWANT REFERENCE")
print("=" * 80)

import kwant
a = 1.42
t = -2.7
graphene = kwant.lattice.honeycomb(a, name='graphene')
a_sublattice, b_sublattice = graphene.sublattices

width = 2
lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((1, 0))))
for iy in range(width):
    lead[a_sublattice(0, iy)] = 0
    lead[b_sublattice(0, iy)] = 0
lead[graphene.neighbors()] = t

sys = kwant.Builder()
for ix in range(3):
    for iy in range(width):
        sys[a_sublattice(ix, iy)] = 0
        sys[b_sublattice(ix, iy)] = 0
sys[graphene.neighbors()] = t
sys.attach_lead(lead)
sys.attach_lead(lead.reversed())

fsys = sys.finalized()
H00_kwant = fsys.leads[0].cell_hamiltonian(sparse=False)

print(f"\nKWANT H00 (4×4):\n{H00_kwant}")
print(f"\ndet(H00_kwant) = {np.linalg.det(H00_kwant):.3e}")
print(f"Rank of H00_kwant: {np.linalg.matrix_rank(H00_kwant)}/4")

print("\nKWANT Connectivity:")
for i in range(4):
    neighbors = np.where(np.abs(H00_kwant[i, :]) > 0.1)[0]
    print(f"  Atom {i}: connects to {neighbors.tolist()}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"My H00 matches KWANT? {np.allclose(H00, H00_kwant)}")
if not np.allclose(H00, H00_kwant):
    print(f"Max difference: {np.max(np.abs(H00 - H00_kwant)):.6e}")
    print("\nDifference matrix:")
    print(H00 - H00_kwant)
