"""
Understand graphene zigzag periodicity properly.
"""
import numpy as np
import matplotlib.pyplot as plt

# Graphene lattice vectors
a = 1.42  # Carbon-carbon distance (Å)
a1 = np.array([np.sqrt(3) * a, 0])
a2 = np.array([np.sqrt(3) * a / 2, 3 * a / 2])

# For zigzag ribbon, transport is along x
# Unit cell contains one row of zigzag edge
# Let's place 4 atoms: 2 zigzag chains, each with A and B sublattices

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Show 3 unit cells of a width-2 zigzag ribbon
positions_per_cell = []
for cell_idx in range(3):
    x_offset = cell_idx * np.sqrt(3) * a
    # Chain 0
    pos_A0 = np.array([x_offset, 0])
    pos_B0 = np.array([x_offset, a])
    # Chain 1
    pos_A1 = np.array([x_offset, 3 * a])
    pos_B1 = np.array([x_offset, 4 * a])
    
    positions_per_cell.append([pos_A0, pos_B0, pos_A1, pos_B1])
    
    colors = ['red' if cell_idx == 0 else 'blue']
    marker = 'o' if cell_idx == 0 else 's'
    size = 150 if cell_idx == 0 else 100
    
    for i, pos in enumerate([pos_A0, pos_B0, pos_A1, pos_B1]):
        ax1.scatter(pos[0], pos[1], c=colors, s=size, marker=marker, zorder=3)
        ax1.text(pos[0], pos[1] + 0.4, f"{cell_idx}:{i}", ha='center', fontsize=9)

# Draw bonds
t = -2.7  # eV
# Intra-cell bonds (H00 for cell 0)
cell0 = positions_per_cell[0]
# A0-B0
ax1.plot([cell0[0][0], cell0[1][0]], [cell0[0][1], cell0[1][1]], 'k-', lw=2, alpha=0.8, label='H00')
# B0-A1
ax1.plot([cell0[1][0], cell0[2][0]], [cell0[1][1], cell0[2][1]], 'k-', lw=2, alpha=0.8)
# A1-B1
ax1.plot([cell0[2][0], cell0[3][0]], [cell0[2][1], cell0[3][1]], 'k-', lw=2, alpha=0.8)
# A0-B1 (diagonal across)
dx = np.sqrt(3) * a / 2
dy = 3.5 * a
dist = np.sqrt(dx**2 + dy**2)
if abs(dist - a) < 0.1:  # Check if it's actually a bond
    ax1.plot([cell0[0][0], cell0[3][0]], [cell0[0][1], cell0[3][1]], 'k--', lw=1, alpha=0.5, label='Long diagonal?')

# Inter-cell bonds (H01: cell 0 → cell 1)
cell1 = positions_per_cell[1]
# A0 → A0 (horizontal)
ax1.plot([cell0[0][0], cell1[0][0]], [cell0[0][1], cell1[0][1]], 'g-', lw=2, alpha=0.8, label='H01')
# B0 → B0 (horizontal)
ax1.plot([cell0[1][0], cell1[1][0]], [cell0[1][1], cell1[1][1]], 'g-', lw=2, alpha=0.8)
# A1 → A1 (horizontal)
ax1.plot([cell0[2][0], cell1[2][0]], [cell0[2][1], cell1[2][1]], 'g-', lw=2, alpha=0.8)
# B1 → B1 (horizontal)
ax1.plot([cell0[3][0], cell1[3][0]], [cell0[3][1], cell1[3][1]], 'g-', lw=2, alpha=0.8)

# Check for diagonal inter-cell bonds
# A0 → B1 of next cell?
dist = np.linalg.norm(cell1[3] - cell0[0])
print(f"Distance A0(cell0) to B1(cell1): {dist:.3f} Å (should be {a:.3f} for bond)")
if abs(dist - a) < 0.1:
    ax1.plot([cell0[0][0], cell1[3][0]], [cell0[0][1], cell1[3][1]], 'g--', lw=1.5, alpha=0.6, label='Diagonal H01')

# B1 → A0 of next cell?
dist = np.linalg.norm(cell1[0] - cell0[3])
print(f"Distance B1(cell0) to A0(cell1): {dist:.3f} Å (should be {a:.3f} for bond)")
if abs(dist - a) < 0.1:
    ax1.plot([cell0[3][0], cell1[0][0]], [cell0[3][1], cell1[0][1]], 'g--', lw=1.5, alpha=0.6)

ax1.set_xlabel('x (Å)')
ax1.set_ylabel('y (Å)')
ax1.set_title('Graphene Zigzag Ribbon: 3 unit cells (width=2)')
ax1.axis('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right plot: Show the expected H00 and H01 matrices
ax2.text(0.1, 0.9, 'Expected Matrices (4×4 for width=2):', transform=ax2.transAxes, fontsize=12, weight='bold')

# Expected H00
H00_expected = np.array([
    [0, -t, 0, 0],    # A0: connects to B0
    [-t, 0, -t, 0],   # B0: connects to A0, A1
    [0, -t, 0, -t],   # A1: connects to B0, B1
    [0, 0, -t, 0]     # B1: connects to A1
])

# Expected H01 (only horizontal bonds)
H01_expected = np.array([
    [-t, 0, 0, 0],    # A0 → A0 next
    [0, -t, 0, 0],    # B0 → B0 next
    [0, 0, -t, 0],    # A1 → A1 next
    [0, 0, 0, -t]     # B1 → B1 next
])

txt = f"H00 (intra-cell):\n{H00_expected}\n\n"
txt += f"H01 (inter-cell):\n{H01_expected}\n\n"
txt += f"det(H00) = {np.linalg.det(H00_expected):.1f}\n"
txt += f"det(H01) = {np.linalg.det(H01_expected):.1f}\n\n"
txt += "If det(H01) = 0, lead is singular!"

ax2.text(0.1, 0.1, txt, transform=ax2.transAxes, fontsize=10, family='monospace', verticalalignment='bottom')
ax2.axis('off')

plt.tight_layout()
plt.savefig('graphene_periodicity.png', dpi=150)
print("\nSaved to graphene_periodicity.png")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"H01_expected determinant: {np.linalg.det(H01_expected):.3f}")
print("This is diagonal, so det(H01) = product of diagonal = (-t)^4 = t^4 ≠ 0")
print("But wait - if all bonds are just horizontal A→A, B→B,...")
print("Then H01 is indeed diagonal and non-singular!")
print()
print("The issue in our code: we're adding EXTRA bonds that make rows identical!")
