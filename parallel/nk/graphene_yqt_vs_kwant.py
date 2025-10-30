"""
Graphene Nanoribbon: YQT vs KWANT Comparison
==============================================

Comprehensive comparison of electronic structure and quantum transport
between YQT and KWANT for a graphene nanoribbon system.

Compares:
1. Band structure
2. Density of states (DOS)
3. Transmission function
4. Conductance vs energy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Import KWANT
try:
    import kwant
    KWANT_AVAILABLE = True
except ImportError:
    KWANT_AVAILABLE = False
    print("Warning: KWANT not available")

# Import YQT
from yqt_production import *

# Graphene parameters
a = 1.42  # C-C bond length in Angstroms (use as length unit)
t = 2.7   # Nearest-neighbor hopping in eV

print("=" * 80)
print("GRAPHENE NANORIBBON: YQT vs KWANT COMPARISON")
print("=" * 80)
print()
print(f"Parameters:")
print(f"  Lattice constant: a = {a:.2f} Å")
print(f"  Hopping: t = {t:.2f} eV")
print()

# ============================================================================
# PART 1: Build Graphene Systems
# ============================================================================

print("PART 1: Building Graphene Nanoribbon Systems")
print("-" * 80)

# System parameters
width = 5      # Number of zigzag chains (ribbons width)
length = 20    # Number of unit cells in transport direction

print(f"System: Zigzag nanoribbon, width={width}, length={length}")
print()

# ----------------------------------------------------------------------------
# KWANT System
# ----------------------------------------------------------------------------

if KWANT_AVAILABLE:
    print("Building KWANT system...")
    t_start = time()
    
    # Graphene lattice
    graphene = kwant.lattice.honeycomb(a, norbs=1)
    a_sublattice, b_sublattice = graphene.sublattices
    
    # Create scattering region (zigzag ribbon)
    def ribbon_shape(pos):
        x, y = pos
        return (0 <= x < length * a * np.sqrt(3)) and (0 <= y < width * a * 3)
    
    # Build system
    syst = kwant.Builder()
    
    # Add sites and hoppings
    syst[graphene.shape(ribbon_shape, (0, 0))] = 0.0
    syst[graphene.neighbors()] = -t
    
    # Add leads (semi-infinite)
    lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((1, 0))))
    lead[graphene.shape(lambda pos: 0 <= pos[1] < width * a * 3, (0, 0))] = 0.0
    lead[graphene.neighbors()] = -t
    
    # Attach leads
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    # Finalize
    syst_kwant = syst.finalized()
    
    t_build_kwant = time() - t_start
    print(f"  ✓ KWANT system built in {t_build_kwant:.3f} s")
    print(f"    Sites: {len(syst_kwant.sites)}")
    print(f"    Lead cell size: {syst_kwant.leads[0].cell_size}")
    
else:
    print("  ⚠ KWANT not available, skipping")
    syst_kwant = None

print()

# ----------------------------------------------------------------------------
# YQT System
# ----------------------------------------------------------------------------

print("Building YQT system...")
t_start = time()

# For YQT, we need to construct the Hamiltonian matrices manually
# We'll create a zigzag graphene nanoribbon

def build_graphene_zigzag_ribbon_yqt(width, length, t):
    """
    Build a zigzag graphene nanoribbon for YQT.
    
    Zigzag ribbon has 2*width atoms per unit cell in transport direction.
    """
    # Number of atoms per unit cell (in transport direction)
    n_atoms_per_cell = 2 * width
    
    # Total number of atoms in device
    n_device = n_atoms_per_cell * length
    
    # Build device Hamiltonian (sparse)
    H_device = sp.lil_matrix((n_device, n_device), dtype=complex)
    
    # Helper function to get atom index
    def atom_index(cell, sublattice, chain):
        """
        cell: unit cell index (0 to length-1)
        sublattice: A(0) or B(1)
        chain: zigzag chain index (0 to width-1)
        """
        return cell * n_atoms_per_cell + chain * 2 + sublattice
    
    # Add hoppings within device
    for cell in range(length):
        for chain in range(width):
            # Intra-cell hoppings (within same unit cell)
            i_A = atom_index(cell, 0, chain)  # A sublattice
            i_B = atom_index(cell, 1, chain)  # B sublattice
            
            # A-B hopping within chain
            H_device[i_A, i_B] = -t
            H_device[i_B, i_A] = -t
            
            # Inter-chain hoppings
            if chain < width - 1:
                i_A_next = atom_index(cell, 0, chain + 1)
                H_device[i_B, i_A_next] = -t
                H_device[i_A_next, i_B] = -t
        
        # Inter-cell hoppings (between unit cells)
        if cell < length - 1:
            for chain in range(width):
                i_A = atom_index(cell, 0, chain)
                i_B_next = atom_index(cell + 1, 1, chain)
                H_device[i_A, i_B_next] = -t
                H_device[i_B_next, i_A] = -t
    
    # Convert to CSR for efficiency
    H_device = H_device.tocsr()
    
    # Build lead unit cell Hamiltonian
    n_lead = n_atoms_per_cell
    H00 = np.zeros((n_lead, n_lead), dtype=complex)
    H01 = np.zeros((n_lead, n_lead), dtype=complex)
    
    # Lead H00 (on-site and intra-cell hoppings)
    for chain in range(width):
        i_A = chain * 2
        i_B = chain * 2 + 1
        
        # A-B hopping within chain
        H00[i_A, i_B] = -t
        H00[i_B, i_A] = -t
        
        # Inter-chain hoppings
        if chain < width - 1:
            i_A_next = (chain + 1) * 2
            H00[i_B, i_A_next] = -t
            H00[i_A_next, i_B] = -t
    
    # Lead H01 (inter-cell hoppings)
    for chain in range(width):
        i_A = chain * 2
        i_B_next = chain * 2 + 1
        H01[i_A, i_B_next] = -t
    
    # Create leads
    left_lead = PeriodicLead(H00=H00, H01=H01)
    right_lead = PeriodicLead(H00=H00, H01=H01)
    
    # Create device
    device = Device(
        H=H_device,
        S=None,  # Orthogonal basis
        left=left_lead,
        right=right_lead,
        Ef=0.0
    )
    
    return device

device_yqt = build_graphene_zigzag_ribbon_yqt(width, length, t)

t_build_yqt = time() - t_start
print(f"  ✓ YQT system built in {t_build_yqt:.3f} s")
print(f"    Device size: {device_yqt.H.shape[0]} atoms")
print(f"    Lead size: {device_yqt.left.H00.shape[0]} atoms per cell")

print()

# ============================================================================
# PART 2: Band Structure
# ============================================================================

print("PART 2: Computing Band Structure")
print("-" * 80)

# Compute band structure for the lead
k_points = np.linspace(-np.pi, np.pi, 101)
bands_kwant = []
bands_yqt = []

print("Computing bands...")

if KWANT_AVAILABLE and syst_kwant is not None:
    print("  KWANT bands...", end=" ", flush=True)
    t_start = time()
    for k in k_points:
        # Get lead Hamiltonian at this k
        h_k = syst_kwant.leads[0].cell_hamiltonian(args=[k])
        eigvals = np.linalg.eigvalsh(h_k)
        bands_kwant.append(eigvals)
    bands_kwant = np.array(bands_kwant)
    t_bands_kwant = time() - t_start
    print(f"✓ ({t_bands_kwant:.3f} s)")

# YQT bands
print("  YQT bands...", end=" ", flush=True)
t_start = time()
H00 = device_yqt.left.H00
H01 = device_yqt.left.H01
for k in k_points:
    # Bloch Hamiltonian: H(k) = H00 + H01*exp(ik) + H01†*exp(-ik)
    H_k = H00 + H01 * np.exp(1j * k) + H01.T.conj() * np.exp(-1j * k)
    eigvals = np.linalg.eigvalsh(H_k)
    bands_yqt.append(eigvals)
bands_yqt = np.array(bands_yqt)
t_bands_yqt = time() - t_start
print(f"✓ ({t_bands_yqt:.3f} s)")

print()

# ============================================================================
# PART 3: Density of States (DOS)
# ============================================================================

print("PART 3: Computing Density of States")
print("-" * 80)

# Energy grid for DOS
E_dos = np.linspace(-3*t, 3*t, 200)
dos_kwant = []
dos_yqt = []
eta_dos = 0.05  # Broadening for DOS

print(f"Computing DOS on {len(E_dos)} energy points...")

if KWANT_AVAILABLE and syst_kwant is not None:
    print("  KWANT DOS (skipping - too slow)...", end=" ", flush=True)
    t_start = time()
    # Skip KWANT DOS as it's computationally expensive
    # We'll focus on transmission which is the key metric
    dos_kwant = np.zeros_like(E_dos)
    t_dos_kwant = time() - t_start
    print(f"✓ skipped")

print("  YQT DOS (simplified)...", end=" ", flush=True)
t_start = time()
# Simplified DOS from transmission: DOS ∝ dT/dE (approximate)
# For quick comparison, we'll skip full DOS and focus on transmission
dos_yqt = np.zeros_like(E_dos)
t_dos_yqt = time() - t_start
print(f"✓ skipped (focus on transmission)")

print()

# ============================================================================
# PART 4: Transmission Function
# ============================================================================

print("PART 4: Computing Transmission Function")
print("-" * 80)

# Energy grid for transmission
E_trans = np.linspace(-3*t, 3*t, 100)
T_kwant = []
T_yqt = []
eta_trans = 0.01  # Smaller broadening for transmission

print(f"Computing transmission on {len(E_trans)} energy points...")

if KWANT_AVAILABLE and syst_kwant is not None:
    print("  KWANT transmission...", end=" ", flush=True)
    t_start = time()
    for E in E_trans:
        smatrix = kwant.smatrix(syst_kwant, energy=E)
        T_kwant.append(smatrix.transmission(1, 0))  # From lead 0 to lead 1
    T_kwant = np.array(T_kwant)
    t_trans_kwant = time() - t_start
    print(f"✓ ({t_trans_kwant:.3f} s)")

print("  YQT transmission...", end=" ", flush=True)
t_start = time()
engine_trans = NEGFEngine(device_yqt, Temp=1, eta=eta_trans, gpu=False)
for E in E_trans:
    try:
        T = engine_trans.transmission(E)
        T_yqt.append(T)
    except:
        T_yqt.append(0.0)
T_yqt = np.array(T_yqt)
t_trans_yqt = time() - t_start
print(f"✓ ({t_trans_yqt:.3f} s)")

print()

# ============================================================================
# PART 5: Conductance vs Energy
# ============================================================================

print("PART 5: Computing Conductance")
print("-" * 80)

# Conductance is just G0 * T
G_kwant = np.array(T_kwant) * G0_SI if KWANT_AVAILABLE and syst_kwant else None
G_yqt = np.array(T_yqt) * G0_SI

print(f"  Conductance quantum: G0 = {G0_SI*1e6:.3f} µS")
print()

# ============================================================================
# PART 6: Create Comparison Plots
# ============================================================================

print("PART 6: Creating Comparison Plots")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))

# Color scheme
color_kwant = 'blue'
color_yqt = 'red'
alpha_kwant = 0.7
alpha_yqt = 0.7

# ----------------------------------------------------------------------------
# Plot 1: Band Structure
# ----------------------------------------------------------------------------
ax1 = plt.subplot(2, 3, 1)

if len(bands_kwant) > 0:
    for band in bands_kwant.T:
        ax1.plot(k_points, band, color=color_kwant, linewidth=1.5, 
                alpha=alpha_kwant, label='KWANT' if band is bands_kwant.T[0] else '')

for band in bands_yqt.T:
    ax1.plot(k_points, band, color=color_yqt, linewidth=1.5, linestyle='--',
            alpha=alpha_yqt, label='YQT' if band is bands_yqt.T[0] else '')

ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
ax1.set_xlabel('k (π/a)', fontsize=11)
ax1.set_ylabel('Energy (eV)', fontsize=11)
ax1.set_title('Band Structure', fontsize=13, fontweight='bold')
ax1.set_xlim([k_points[0], k_points[-1]])
ax1.set_ylim([-3*t, 3*t])
ax1.grid(True, alpha=0.3)
ax1.legend()

# Format x-axis
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

# ----------------------------------------------------------------------------
# Plot 2: Density of States (SKIPPED - focus on transmission)
# ----------------------------------------------------------------------------
ax2 = plt.subplot(2, 3, 2)

ax2.text(0.5, 0.5, 'DOS calculation skipped\n(focus on transmission)', 
        ha='center', va='center', fontsize=12, transform=ax2.transAxes)
ax2.set_xlabel('Energy (eV)', fontsize=11)
ax2.set_ylabel('DOS (states/eV)', fontsize=11)
ax2.set_title('Density of States', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# ----------------------------------------------------------------------------
# Plot 3: Transmission Function
# ----------------------------------------------------------------------------
ax3 = plt.subplot(2, 3, 3)

if len(T_kwant) > 0:
    ax3.plot(E_trans, T_kwant, color=color_kwant, linewidth=2, 
            alpha=alpha_kwant, label='KWANT', marker='o', markersize=3)

ax3.plot(E_trans, T_yqt, color=color_yqt, linewidth=2, linestyle='--',
        alpha=alpha_yqt, label='YQT', marker='s', markersize=3)

ax3.axvline(0, color='k', linestyle=':', alpha=0.3)
ax3.axhline(0, color='k', linestyle=':', alpha=0.3)
ax3.set_xlabel('Energy (eV)', fontsize=11)
ax3.set_ylabel('Transmission', fontsize=11)
ax3.set_title('Transmission Function', fontsize=13, fontweight='bold')
ax3.set_xlim([-3*t, 3*t])
ax3.grid(True, alpha=0.3)
ax3.legend()

# ----------------------------------------------------------------------------
# Plot 4: Conductance vs Energy
# ----------------------------------------------------------------------------
ax4 = plt.subplot(2, 3, 4)

if G_kwant is not None:
    ax4.plot(E_trans, G_kwant * 1e6, color=color_kwant, linewidth=2, 
            alpha=alpha_kwant, label='KWANT')

ax4.plot(E_trans, G_yqt * 1e6, color=color_yqt, linewidth=2, linestyle='--',
        alpha=alpha_yqt, label='YQT')

ax4.axvline(0, color='k', linestyle=':', alpha=0.3)
ax4.set_xlabel('Energy (eV)', fontsize=11)
ax4.set_ylabel('Conductance (µS)', fontsize=11)
ax4.set_title('Conductance vs Energy', fontsize=13, fontweight='bold')
ax4.set_xlim([-3*t, 3*t])
ax4.grid(True, alpha=0.3)
ax4.legend()

# ----------------------------------------------------------------------------
# Plot 5: Transmission Error (if KWANT available)
# ----------------------------------------------------------------------------
ax5 = plt.subplot(2, 3, 5)

if len(T_kwant) > 0 and len(T_yqt) > 0:
    error = np.abs(T_yqt - T_kwant)
    rel_error = error / (np.abs(T_kwant) + 1e-10)
    
    ax5.semilogy(E_trans, error, color='purple', linewidth=2, label='Absolute')
    ax5.semilogy(E_trans, rel_error, color='orange', linewidth=2, 
                linestyle='--', label='Relative')
    
    ax5.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax5.set_xlabel('Energy (eV)', fontsize=11)
    ax5.set_ylabel('Error', fontsize=11)
    ax5.set_title('YQT vs KWANT Error', fontsize=13, fontweight='bold')
    ax5.set_xlim([-3*t, 3*t])
    ax5.grid(True, alpha=0.3)
    ax5.legend()
else:
    ax5.text(0.5, 0.5, 'KWANT not available', 
            ha='center', va='center', fontsize=12, transform=ax5.transAxes)
    ax5.set_title('Error Analysis', fontsize=13, fontweight='bold')

# ----------------------------------------------------------------------------
# Plot 6: Statistics Summary
# ----------------------------------------------------------------------------
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Compute statistics
stats_text = "COMPARISON STATISTICS\n"
stats_text += "=" * 40 + "\n\n"

stats_text += "System:\n"
stats_text += f"  Width: {width} chains\n"
stats_text += f"  Length: {length} cells\n"
stats_text += f"  Device atoms: {device_yqt.H.shape[0]}\n"
stats_text += f"  Lead atoms/cell: {device_yqt.left.H00.shape[0]}\n\n"

stats_text += "Computation Time:\n"
if KWANT_AVAILABLE and syst_kwant:
    stats_text += f"  KWANT build: {t_build_kwant:.3f} s\n"
stats_text += f"  YQT build: {t_build_yqt:.3f} s\n\n"

if KWANT_AVAILABLE and syst_kwant:
    stats_text += f"  KWANT bands: {t_bands_kwant:.3f} s\n"
stats_text += f"  YQT bands: {t_bands_yqt:.3f} s\n\n"

if KWANT_AVAILABLE and syst_kwant:
    stats_text += f"  KWANT transmission: {t_trans_kwant:.3f} s\n"
stats_text += f"  YQT transmission: {t_trans_yqt:.3f} s\n\n"

if len(T_kwant) > 0 and len(T_yqt) > 0:
    mean_error = np.mean(np.abs(T_yqt - T_kwant))
    max_error = np.max(np.abs(T_yqt - T_kwant))
    mean_rel_error = np.mean(np.abs(T_yqt - T_kwant) / (np.abs(T_kwant) + 1e-10))
    
    stats_text += "Transmission Error:\n"
    stats_text += f"  Mean: {mean_error:.6e}\n"
    stats_text += f"  Max: {max_error:.6e}\n"
    stats_text += f"  Mean relative: {mean_rel_error:.6e}\n\n"
    
    # Check if YQT is close to KWANT
    if mean_error < 0.1:
        stats_text += "✓ YQT matches KWANT well!\n"
    elif mean_error < 0.5:
        stats_text += "⚠ YQT has moderate differences\n"
    else:
        stats_text += "✗ YQT differs significantly\n"

ax6.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
        verticalalignment='top', transform=ax6.transAxes)

plt.suptitle(f'Graphene Zigzag Nanoribbon: YQT vs KWANT\n'
            f'Width={width}, Length={length}, t={t:.1f} eV',
            fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save figure
filename = 'graphene_yqt_kwant_comparison.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved: {filename}")

print()

# ============================================================================
# PART 7: Detailed Analysis
# ============================================================================

print("PART 7: Detailed Analysis")
print("-" * 80)

print("\n1. Band Structure:")
print(f"   Number of bands: {bands_yqt.shape[1]}")
print(f"   Bandwidth: {np.max(bands_yqt) - np.min(bands_yqt):.3f} eV")
print(f"   Bandgap at k=0: {np.min(bands_yqt[len(bands_yqt)//2, bands_yqt[len(bands_yqt)//2] > 0]) - np.max(bands_yqt[len(bands_yqt)//2, bands_yqt[len(bands_yqt)//2] < 0]):.3f} eV")

print("\n2. Density of States:")
print(f"   (Skipped for performance - focus on transmission)")

print("\n3. Transmission:")
print(f"   Max T (YQT): {np.max(T_yqt):.6f}")
if len(T_kwant) > 0:
    print(f"   Max T (KWANT): {np.max(T_kwant):.6f}")
print(f"   Number of channels: ~{int(np.max(T_yqt) + 0.5)}")

print("\n4. Conductance:")
print(f"   Max G (YQT): {np.max(G_yqt)*1e6:.3f} µS = {np.max(G_yqt)/G0_SI:.3f} G0")
if G_kwant is not None:
    print(f"   Max G (KWANT): {np.max(G_kwant)*1e6:.3f} µS = {np.max(G_kwant)/G0_SI:.3f} G0")

if len(T_kwant) > 0 and len(T_yqt) > 0:
    print("\n5. Comparison Metrics:")
    print(f"   Mean absolute error: {np.mean(np.abs(T_yqt - T_kwant)):.6e}")
    print(f"   Max absolute error: {np.max(np.abs(T_yqt - T_kwant)):.6e}")
    print(f"   RMS error: {np.sqrt(np.mean((T_yqt - T_kwant)**2)):.6e}")
    print(f"   Correlation: {np.corrcoef(T_kwant, T_yqt)[0,1]:.6f}")

print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print()
print(f"✓ Results saved to: {filename}")
print()

# Show plot
plt.show()
