"""
Extract KWANT's graphene positions and use them in our builder.
"""
import kwant
import numpy as np

a = 1.42
graphene = kwant.lattice.honeycomb(a, name='graphene')
a_sublattice, b_sublattice = graphene.sublattices

width = 3  # Try different widths

# Get positions from KWANT
print("=" * 80)
print(f"KWANT GRAPHENE POSITIONS (width={width})")
print("=" * 80)

for iy in range(width):
    pos_A = a_sublattice(0, iy).pos
    pos_B = b_sublattice(0, iy).pos
    print(f"Chain {iy}:")
    print(f"  A[0,{iy}]: pos = [{pos_A[0]:.6f}, {pos_A[1]:.6f}]")
    print(f"  B[0,{iy}]: pos = [{pos_B[0]:.6f}, {pos_B[1]:.6f}]")
    
# Get primitive vectors
print("\n" + "=" * 80)
print("LATTICE VECTORS")
print("=" * 80)
prim_vecs = graphene.prim_vecs
print(f"a1 (transport direction): {prim_vecs[0]}")
print(f"a2 (transverse direction): {prim_vecs[1]}")

# Sublattice offsets
print("\nSublattice positions within primitive cell:")
for sublat in graphene.sublattices:
    print(f"  {sublat.name}: offset = {sublat.offset}")

print("\n" + "=" * 80)
print("POSITION FORMULA")
print("=" * 80)
print("For site (ix, iy) in sublattice with offset r_sub:")
print("  pos = ix * a1 + iy * a2 + r_sub")
print()
print("For honeycomb:")
print(f"  a1 = sqrt(3)*a * [1, 0] = {np.sqrt(3)*a:.6f} * [1, 0]")
print(f"  a2 = a * [sqrt(3)/2, 3/2] = {a:.6f} * [{np.sqrt(3)/2:.6f}, {3/2:.6f}]")
print(f"  r_A = [0, 0]")
print(f"  r_B = [0, a] = [0, {a:.6f}]")

# Verify
print("\nVerification for (0, 1):")
ix, iy = 0, 1
pos_A_calc = ix * prim_vecs[0] + iy * prim_vecs[1] + np.array([0, 0])
pos_B_calc = ix * prim_vecs[0] + iy * prim_vecs[1] + np.array([0, a])
pos_A_kwant = a_sublattice(ix, iy).pos
pos_B_kwant = b_sublattice(ix, iy).pos

print(f"  Calculated A: {pos_A_calc}")
print(f"  KWANT A:      {pos_A_kwant}")
print(f"  Match: {np.allclose(pos_A_calc, pos_A_kwant)}")
print()
print(f"  Calculated B: {pos_B_calc}")
print(f"  KWANT B:      {pos_B_kwant}")
print(f"  Match: {np.allclose(pos_B_calc, pos_B_kwant)}")
