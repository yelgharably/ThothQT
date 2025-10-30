"""
Understand KWANT's site ordering for zigzag ribbons.
"""
import kwant
import numpy as np

a = 1.42
graphene = kwant.lattice.honeycomb(a, name='graphene')
a_sublattice, b_sublattice = graphene.sublattices

width = 2

# Build lead
lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((1, 0))))
for iy in range(width):
    lead[a_sublattice(0, iy)] = 0
    lead[b_sublattice(0, iy)] = 0
lead[graphene.neighbors()] = -2.7

# Build system
sys = kwant.Builder()
for ix in range(3):
    for iy in range(width):
        sys[a_sublattice(ix, iy)] = 0
        sys[b_sublattice(ix, iy)] = 0
sys[graphene.neighbors()] = -2.7
sys.attach_lead(lead)
sys.attach_lead(lead.reversed())

fsys = sys.finalized()

print("=" * 80)
print("KWANT SITE ORDERING")
print("=" * 80)

# Get positions
lead_0 = fsys.leads[0]
print(f"\nLead has {len(lead_0.sites)} sites")

print("\nLead sites (in order):")
for i, site in enumerate(lead_0.sites):
    pos = site.pos
    tag = site.tag
    family = site.family.name
    print(f"  Site {i}: {family} tag={tag}, pos=[{pos[0]:.3f}, {pos[1]:.3f}]")

print("\nDevice sites (first 8):")
for i in range(min(8, len(fsys.sites))):
    site = fsys.sites[i]
    pos = site.pos
    tag = site.tag
    family = site.family.name
    print(f"  Site {i}: {family} tag={tag}, pos=[{pos[0]:.3f}, {pos[1]:.3f}]")

# Get Hamiltonian and show connectivity
H00 = lead_0.cell_hamiltonian(sparse=False)
n_cell = H00.shape[0]

print(f"\nH00 is {n_cell}×{n_cell}")
print(f"Lead has {len(lead_0.sites)} sites total (includes neighboring cells)")
print(f"Unit cell contains sites 0-{n_cell-1}")

print("\n" + "=" * 80)
print("CONNECTIVITY PATTERN (unit cell only)")
print("=" * 80)

for i in range(n_cell):
    neighbors = []
    for j in range(n_cell):
        if abs(H00[i, j]) > 0.1 and i != j:
            site_j = lead_0.sites[j]
            neighbors.append(f"{j}({site_j.family.name}{site_j.tag})")
    print(f"  Site {i} ({lead_0.sites[i].family.name}{lead_0.sites[i].tag}): → {', '.join(neighbors)}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check sublattice pattern (only unit cell)
a_indices = []
b_indices = []
for i in range(n_cell):
    site = lead_0.sites[i]
    if 'graphene0' in site.family.name:  # A sublattice
        a_indices.append(i)
    else:  # B sublattice
        b_indices.append(i)

print(f"A sublattice indices: {a_indices}")
print(f"B sublattice indices: {b_indices}")

# KWANT uses (family, tag) ordering, not (ix, iy, sublattice)
# The order is: all A sites first, then all B sites? Or alternating by tag?
print("\nOrdering scheme:")
if a_indices == [0, 1] and b_indices == [2, 3]:
    print("  KWANT orders: All A sites, then all B sites")
elif a_indices == [0, 2] and b_indices == [1, 3]:
    print("  KWANT orders: Alternating by tag (A0, B0, A1, B1)")
else:
    print(f"  Unknown ordering")
