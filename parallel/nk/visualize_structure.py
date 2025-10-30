"""
Visualize the graphene zigzag ribbon structure to understand the connectivity.
"""
import numpy as np
import matplotlib.pyplot as plt
from toothqt_builders import GrapheneBuilder

# Build a small ribbon
gb = GrapheneBuilder()
device, positions = gb.zigzag_ribbon(width=2, length=3, return_positions=True)

print("=" * 80)
print("GRAPHENE ZIGZAG RIBBON STRUCTURE")
print("=" * 80)
print(f"\nDevice: {device.H.shape[0]} atoms")
print(f"Lead: {device.left.H00.shape[0]} atoms/cell")
print()

# Get the Hamiltonians
H = device.H.toarray()
H00 = device.left.H00
H01 = device.left.H01

n_lead = H00.shape[0]
n_device = H.shape[0]

print("LEAD STRUCTURE")
print("-" * 80)
print(f"H00 (intra-cell):\n{H00}")
print(f"\nH01 (inter-cell):\n{H01}")

print("\nDEVICE STRUCTURE")
print("-" * 80)
print(f"First {n_lead} atoms (should match lead structure):")
print(f"H[0:{n_lead}, 0:{n_lead}]:\n{H[0:n_lead, 0:n_lead]}")

print(f"\nCoupling to next cell:")
print(f"H[0:{n_lead}, {n_lead}:{2*n_lead}]:\n{H[0:n_lead, n_lead:2*n_lead]}")

print("\nVISUAL CHECK:")
print("-" * 80)
print("Does H[0:n_lead, 0:n_lead] match H00?", np.allclose(H[0:n_lead, 0:n_lead], H00))
print("Does H[0:n_lead, n_lead:2*n_lead] match H01?", np.allclose(H[0:n_lead, n_lead:2*n_lead], H01))

# Visualize atom positions
fig, ax = plt.subplots(figsize=(12, 6))

# Color first unit cell differently
colors = ['red' if i < n_lead else 'blue' for i in range(n_device)]
ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=100, zorder=3)

# Add atom labels
for i, pos in enumerate(positions):
    ax.text(pos[0], pos[1] + 0.3, str(i), ha='center', fontsize=8)

# Draw bonds from Hamiltonian
for i in range(n_device):
    for j in range(i+1, n_device):
        if abs(H[i, j]) > 0.1:  # There's a hopping
            ax.plot([positions[i, 0], positions[j, 0]], 
                   [positions[i, 1], positions[j, 1]], 
                   'k-', alpha=0.3, linewidth=0.5)

# Mark unit cell boundaries
x_cells = [i * gb.a * np.sqrt(3) for i in range(4)]
for x in x_cells:
    ax.axvline(x, color='green', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.set_title('Graphene Zigzag Ribbon (red=lead region, blue=device)')
ax.axis('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphene_structure.png', dpi=150)
print("\nSaved structure visualization to graphene_structure.png")

# Check connectivity pattern
print("\nCONNECTIVITY ANALYSIS:")
print("-" * 80)
print("Atoms 0-3 (first unit cell, should be lead-like):")
for i in range(min(4, n_device)):
    neighbors = np.where(np.abs(H[i, :]) > 0.1)[0]
    print(f"  Atom {i}: connects to {neighbors.tolist()}")

print("\nExpected from H00:")
for i in range(n_lead):
    neighbors = np.where(np.abs(H00[i, :]) > 0.1)[0]
    print(f"  Lead atom {i}: connects to {neighbors.tolist()} (in same cell)")

print("\nExpected from H01:")
for i in range(n_lead):
    neighbors = np.where(np.abs(H01[i, :]) > 0.1)[0]
    print(f"  Lead atom {i}: connects to {neighbors.tolist()} (in next cell)")

print("=" * 80)
