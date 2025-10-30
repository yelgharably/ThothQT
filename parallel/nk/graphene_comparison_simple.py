"""
Graphene Nanoribbon: YQT vs KWANT - Simple Comparison
======================================================

Direct comparison using consistent graphene armchair nanoribbon.
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

# Import YQT
from yqt_production import *

# Graphene parameters
a = 1.0  # Lattice constant (normalized)
t = 1.0  # Hopping parameter (energy unit)

print("=" * 80)
print("GRAPHENE ARMCHAIR NANORIBBON: YQT vs KWANT")
print("=" * 80)
print()

# ============================================================================
# Build Consistent Systems
# ============================================================================

# Armchair nanoribbon parameters
width = 3    # Width in dimers
length = 10  # Length in unit cells

print(f"System: Armchair nanoribbon")
print(f"  Width: {width} dimers = {2*width} atoms/cell")
print(f"  Length: {length} cells")
print(f"  Total: {2*width*length} atoms")
print()

# ----------------------------------------------------------------------------
# KWANT System
# ----------------------------------------------------------------------------

if KWANT_AVAILABLE:
    print("Building KWANT system...")
    t_start = time()
    
    # Graphene lattice
    lat = kwant.lattice.honeycomb(a, norbs=1)
    syst = kwant.Builder()
    
    # Armchair ribbon shape
    def armchair_ribbon(pos):
        x, y = pos
        return (0 <= y < width) and (0 <= x < length * 3**0.5 * a)
    
    syst[lat.shape(armchair_ribbon, (0, 0))] = 0
    syst[lat.neighbors()] = -t
    
    # Lead
    lead = kwant.Builder(kwant.TranslationalSymmetry([3**0.5 * a, 0]))
    lead[lat.shape(lambda pos: 0 <= pos[1] < width, (0, 0))] = 0
    lead[lat.neighbors()] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    fsyst_kwant = syst.finalized()
    
    t_kwant_build = time() - t_start
    print(f"  ✓ Built in {t_kwant_build:.3f} s")
    print(f"    Device sites: {len(fsyst_kwant.sites)}")
    print(f"    Lead cell size: {fsyst_kwant.leads[0].cell_size}")
else:
    fsyst_kwant = None
    t_kwant_build = 0

print()

# ----------------------------------------------------------------------------
# YQT System - Match KWANT exactly
# ----------------------------------------------------------------------------

print("Building YQT system...")
t_start = time()

# Build armchair ribbon that matches KWANT
# Armchair: 2 atoms per unit cell in transport direction
n_width = 2 * width  # Total atoms across width
n_device = n_width * length

# Device Hamiltonian
H_dev = sp.lil_matrix((n_device, n_device), dtype=complex)

# Build device Hamiltonian (armchair structure)
for ix in range(length):
    for iy in range(width):
        # Dimer indices in this unit cell
        i1 = ix * n_width + iy * 2
        i2 = ix * n_width + iy * 2 + 1
        
        # Intra-dimer hopping
        H_dev[i1, i2] = -t
        H_dev[i2, i1] = -t
        
        # Inter-dimer hoppings (vertical)
        if iy < width - 1:
            i3 = ix * n_width + (iy + 1) * 2
            H_dev[i1, i3] = -t
            H_dev[i3, i1] = -t
            H_dev[i2, i3] = -t
            H_dev[i3, i2] = -t
        
        # Inter-cell hoppings (horizontal)
        if ix < length - 1:
            j1 = (ix + 1) * n_width + iy * 2
            j2 = (ix + 1) * n_width + iy * 2 + 1
            H_dev[i1, j1] = -t
            H_dev[j1, i1] = -t
            H_dev[i2, j2] = -t
            H_dev[j2, i2] = -t

H_dev = H_dev.tocsr()

# Lead Hamiltonian (one unit cell)
H00 = np.zeros((n_width, n_width), dtype=complex)
H01 = np.zeros((n_width, n_width), dtype=complex)

for iy in range(width):
    i1 = iy * 2
    i2 = iy * 2 + 1
    
    # Intra-cell: dimer hopping
    H00[i1, i2] = -t
    H00[i2, i1] = -t
    
    # Intra-cell: vertical hoppings
    if iy < width - 1:
        i3 = (iy + 1) * 2
        H00[i1, i3] = -t
        H00[i3, i1] = -t
        H00[i2, i3] = -t
        H00[i3, i2] = -t
    
    # Inter-cell: horizontal hoppings
    H01[i1, i1] = -t
    H01[i2, i2] = -t

# Create device
lead_left = PeriodicLead(H00=H00, H01=H01)
lead_right = PeriodicLead(H00=H00, H01=H01)
device_yqt = Device(H=H_dev, S=None, left=lead_left, right=lead_right, Ef=0.0)

t_yqt_build = time() - t_start
print(f"  ✓ Built in {t_yqt_build:.3f} s")
print(f"    Device sites: {n_device}")
print(f"    Lead cell size: {n_width}")

print()

# ============================================================================
# Compare Band Structures
# ============================================================================

print("Computing Band Structure...")
print("-" * 80)

k_vals = np.linspace(-np.pi, np.pi, 101)

# KWANT bands
if KWANT_AVAILABLE and fsyst_kwant:
    print("  KWANT...", end=" ", flush=True)
    t_start = time()
    bands_kwant = []
    H00_kwant = H01_kwant = None
    try:
        # Get lead Hamiltonian matrices
        H00_kwant = fsyst_kwant.leads[0].cell_hamiltonian()
        H01_kwant = fsyst_kwant.leads[0].inter_cell_hopping()
        
        for k in k_vals:
            H_k = H00_kwant + H01_kwant * np.exp(1j * k) + H01_kwant.T.conj() * np.exp(-1j * k)
            eigvals = np.linalg.eigvalsh(H_k)
            bands_kwant.append(eigvals)
        bands_kwant = np.array(bands_kwant)
        t_kwant_bands = time() - t_start
        print(f"✓ {t_kwant_bands:.3f} s")
    except Exception as e:
        print(f"✗ Error: {e}")
        bands_kwant = None
else:
    bands_kwant = None

# YQT bands
print("  YQT...", end=" ", flush=True)
t_start = time()
bands_yqt = []
for k in k_vals:
    H_k = H00 + H01 * np.exp(1j * k) + H01.T.conj() * np.exp(-1j * k)
    eigvals = np.linalg.eigvalsh(H_k)
    bands_yqt.append(eigvals)
bands_yqt = np.array(bands_yqt)
t_yqt_bands = time() - t_start
print(f"✓ {t_yqt_bands:.3f} s")

print()

# ============================================================================
# Compare Transmission
# ============================================================================

print("Computing Transmission...")
print("-" * 80)

E_vals = np.linspace(-2.5, 2.5, 80)
eta_val = 0.02

# KWANT transmission
if KWANT_AVAILABLE and fsyst_kwant:
    print(f"  KWANT ({len(E_vals)} points)...", end=" ", flush=True)
    t_start = time()
    T_kwant = []
    for E in E_vals:
        try:
            smat = kwant.smatrix(fsyst_kwant, E)
            T_kwant.append(smat.transmission(1, 0))
        except:
            T_kwant.append(0.0)
    T_kwant = np.array(T_kwant)
    t_kwant_trans = time() - t_start
    print(f"✓ {t_kwant_trans:.3f} s")
else:
    T_kwant = None

# YQT transmission
print(f"  YQT ({len(E_vals)} points)...", end=" ", flush=True)
t_start = time()
engine = NEGFEngine(device_yqt, Temp=1, eta=eta_val, gpu=False)
T_yqt = []
for E in E_vals:
    try:
        T = engine.transmission(E)
        T_yqt.append(T)
    except:
        T_yqt.append(0.0)
T_yqt = np.array(T_yqt)
t_yqt_trans = time() - t_start
print(f"✓ {t_yqt_trans:.3f} s")

print()

# ============================================================================
# Analysis & Plots
# ============================================================================

print("Analysis:")
print("-" * 80)

if T_kwant is not None:
    print(f"\nTransmission:")
    print(f"  KWANT: [{np.min(T_kwant):.4f}, {np.max(T_kwant):.4f}]")
    print(f"  YQT:   [{np.min(T_yqt):.4f}, {np.max(T_yqt):.4f}]")
    
    error = np.abs(T_yqt - T_kwant)
    print(f"\nError:")
    print(f"  Mean: {np.mean(error):.6f}")
    print(f"  Max:  {np.max(error):.6f}")
    print(f"  RMS:  {np.sqrt(np.mean(error**2)):.6f}")
    
    corr = np.corrcoef(T_kwant, T_yqt)[0, 1]
    print(f"  Correlation: {corr:.6f}")

print(f"\nBands:")
print(f"  Number: {bands_yqt.shape[1]}")
print(f"  Range: [{np.min(bands_yqt):.3f}, {np.max(bands_yqt):.3f}] eV")

print()

# ============================================================================
# Create Plots
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Band structure
ax = axes[0, 0]
if bands_kwant is not None:
    for band in bands_kwant.T:
        ax.plot(k_vals, band, 'b-', linewidth=1.5, alpha=0.6)
for band in bands_yqt.T:
    line_style = '--' if bands_kwant is not None else '-'
    ax.plot(k_vals, band, 'r' + line_style, linewidth=1.5, alpha=0.8)
ax.axhline(0, color='k', linestyle=':', alpha=0.3)
ax.set_xlabel('k (π/a)', fontsize=11)
ax.set_ylabel('Energy (t)', fontsize=11)
ax.set_title('Band Structure', fontsize=12, fontweight='bold')
ax.set_xlim([k_vals[0], k_vals[-1]])
ax.grid(True, alpha=0.3)
if bands_kwant is not None:
    ax.legend(['KWANT', 'YQT'])
else:
    ax.legend(['YQT'])

# Transmission
ax = axes[0, 1]
if T_kwant is not None:
    ax.plot(E_vals, T_kwant, 'b-', linewidth=2, label='KWANT', marker='o', markersize=3)
ax.plot(E_vals, T_yqt, 'r--', linewidth=2, label='YQT', marker='s', markersize=3)
ax.axvline(0, color='k', linestyle=':', alpha=0.3)
ax.axhline(0, color='k', linestyle=':', alpha=0.3)
ax.set_xlabel('Energy (t)', fontsize=11)
ax.set_ylabel('Transmission', fontsize=11)
ax.set_title('Transmission Function', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Conductance
ax = axes[1, 0]
if T_kwant is not None:
    ax.plot(E_vals, T_kwant * G0_SI * 1e6, 'b-', linewidth=2, label='KWANT')
ax.plot(E_vals, T_yqt * G0_SI * 1e6, 'r--', linewidth=2, label='YQT')
ax.axvline(0, color='k', linestyle=':', alpha=0.3)
ax.set_xlabel('Energy (t)', fontsize=11)
ax.set_ylabel('Conductance (µS)', fontsize=11)
ax.set_title('Conductance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Error analysis
ax = axes[1, 1]
if T_kwant is not None:
    error = np.abs(T_yqt - T_kwant)
    ax.semilogy(E_vals, error + 1e-10, 'purple', linewidth=2)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Energy (t)', fontsize=11)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('YQT Error vs KWANT', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No KWANT reference', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)

plt.suptitle(f'Graphene Armchair Nanoribbon: YQT vs KWANT\nWidth={width} dimers, Length={length} cells',
            fontsize=14, fontweight='bold')
plt.tight_layout()

filename = 'graphene_yqt_kwant_simple.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {filename}")

print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

plt.show()
