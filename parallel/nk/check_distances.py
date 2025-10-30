"""
Check actual distances in the honeycomb lattice.
"""
import numpy as np

a = 1.42
a1 = np.array([a, 0])
a2 = np.array([a/2, np.sqrt(3)*a/2])
r_A = np.array([0, 0])
r_B = np.array([0, a/np.sqrt(3)])

width = 2

# Lead positions
lead_pos = {}
for iy in range(width):
    i_A = width - 1 - iy
    i_B = width + iy
    lead_pos[i_A] = 0 * a1 + iy * a2 + r_A
    lead_pos[i_B] = 0 * a1 + iy * a2 + r_B

print("Lead atom positions:")
for i in range(2*width):
    print(f"  Atom {i}: {lead_pos[i]}")

print("\nDistances between all pairs:")
for i in range(2*width):
    for j in range(i+1, 2*width):
        dist = np.linalg.norm(lead_pos[i] - lead_pos[j])
        print(f"  {i}↔{j}: {dist:.6f} (ratio to a: {dist/a:.4f})")

print(f"\nExpected C-C bond length: {a:.6f}")
print(f"Expected 2nd nearest: {a * np.sqrt(3):.6f}")
