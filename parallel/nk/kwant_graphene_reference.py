"""
Build graphene zigzag ribbon using KWANT and extract the matrices.
"""
import kwant
import numpy as np

# Graphene lattice
a = 1.42  # C-C distance
t = -2.7  # Hopping
graphene = kwant.lattice.honeycomb(a, name='graphene')
a_sublattice, b_sublattice = graphene.sublattices

width = 2  # Number of zigzag chains
length = 3  # Device length

# Build system
sys = kwant.Builder()

# Device region
for ix in range(length):
    for iy in range(width):
        sys[a_sublattice(ix, iy)] = 0
        sys[b_sublattice(ix, iy)] = 0

# Add hoppings (KWANT does this automatically for nearest neighbors)
sys[graphene.neighbors()] = t

# Build lead
lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((1, 0))))
for iy in range(width):
    lead[a_sublattice(0, iy)] = 0
    lead[b_sublattice(0, iy)] = 0
lead[graphene.neighbors()] = t

# Attach leads
sys.attach_lead(lead)
sys.attach_lead(lead.reversed())

# Finalize
fsys = sys.finalized()

print("=" * 80)
print("KWANT GRAPHENE ZIGZAG RIBBON")
print("=" * 80)
print(f"Width: {width} zigzag chains")
print(f"Length: {length} unit cells")
print()

# Get device Hamiltonian
H_device = fsys.hamiltonian_submatrix(sparse=False)
print(f"Device Hamiltonian shape: {H_device.shape}")
print(f"Device H:\n{H_device}\n")

# Get lead Hamiltonians
lead_0 = fsys.leads[0]
H00 = lead_0.cell_hamiltonian(sparse=False)
H01 = lead_0.inter_cell_hopping(sparse=False)

print(f"Lead H00 shape: {H00.shape}")
print(f"Lead H00:\n{H00}\n")

print(f"Lead H01 shape: {H01.shape}")
print(f"Lead H01:\n{H01}\n")

# Check properties
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"det(H00) = {np.linalg.det(H00):.3e}")
print(f"H01 shape is {H01.shape} - not square!")
print("KWANT uses minimal unit cell in transport direction.")
print(f"H01 couples {H01.shape[0]} sites in cell n to {H01.shape[1]} sites in cell n+1")

# Band structure
print("\nBand structure check:")
# Can't use H00 + H01 + H01† for non-square H01
# Instead, build Bloch Hamiltonian at k=0
print(f"H00 eigenvalues: {np.linalg.eigvalsh(H00)}")

# Test transmission
E = 0.5  # eV
try:
    smatrix = kwant.smatrix(fsys, E)
    T = smatrix.transmission(1, 0)  # Right lead <- left lead
    print(f"\nTransmission at E={E} eV: {T:.6f}")
except Exception as e:
    print(f"\nTransmission calculation failed: {e}")

print("=" * 80)

# Visualize site ordering
print("\nSITE ORDERING:")
for i, site in enumerate(fsys.sites):
    print(f"  Site {i}: {site}")
