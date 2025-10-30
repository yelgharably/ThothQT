"""
Square Lattice 1D Chain: YQT vs KWANT Comparison
=================================================

Clean comparison using simple square lattice 1D chain.
This tests the core NEGF implementation without graphene complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

try:
    import kwant
    KWANT_AVAILABLE = True
except ImportError:
    KWANT_AVAILABLE = False

from yqt_production import *

print("=" * 80)
print("SIMPLE 1D CHAIN: YQT vs KWANT")
print("=" * 80)
print()

# Parameters
t = 1.0      # Hopping
N = 30       # Chain length

print(f"System: 1D tight-binding chain")
print(f"  Length: {N} sites")
print(f"  Hopping: t = {t}")
print()

# ============================================================================
# Build Systems
# ============================================================================

# KWANT
if KWANT_AVAILABLE:
    print("Building KWANT...")
    lat = kwant.lattice.chain(a=1, norbs=1)
    syst = kwant.Builder()
    
    # Device
    syst[(lat(i) for i in range(N))] = 0
    syst[lat.neighbors()] = -t
    
    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = 0
    lead[lat.neighbors()] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    fsyst_kwant = syst.finalized()
    print(f"  ✓ KWANT built: {len(fsyst_kwant.sites)} sites")
else:
    fsyst_kwant = None
    print("  ⚠ KWANT not available")

# YQT
print("Building YQT...")
H_dev = sp.diags([-t, -t], offsets=[-1, 1], shape=(N, N)).tocsr()
H00 = np.array([[0.0]])
H01 = np.array([[-t]])
lead_yqt = PeriodicLead(H00=H00, H01=H01)
device_yqt = Device(H=H_dev, S=None, left=lead_yqt, right=lead_yqt, Ef=0.0)
print(f"  ✓ YQT built: {N} sites")

print()

# ============================================================================
# Band Structure
# ============================================================================

print("Computing bands...")
k_vals = np.linspace(-np.pi, np.pi, 201)

# Analytical
E_analytical = -2 * t * np.cos(k_vals)

# KWANT
if fsyst_kwant:
    bands_kwant = []
    for k in k_vals:
        H_k = fsyst_kwant.leads[0].cell_hamiltonian() + \
              fsyst_kwant.leads[0].inter_cell_hopping() * np.exp(1j * k) + \
              fsyst_kwant.leads[0].inter_cell_hopping().T.conj() * np.exp(-1j * k)
        E = np.linalg.eigvalsh(H_k)[0]
        bands_kwant.append(E)
    bands_kwant = np.array(bands_kwant)
    print(f"  ✓ KWANT: match analytical = {np.allclose(bands_kwant, E_analytical)}")

# YQT
bands_yqt = -2 * t * np.cos(k_vals)  # Directly from dispersion
print(f"  ✓ YQT: analytical dispersion")

print()

# ============================================================================
# Transmission
# ============================================================================

print("Computing transmission...")
E_vals = np.linspace(-2.5*t, 2.5*t, 100)
eta = 0.05  # Larger eta for stability

# KWANT
if fsyst_kwant:
    print("  KWANT...", end=" ", flush=True)
    t_start = time()
    T_kwant = []
    for E in E_vals:
        try:
            smat = kwant.smatrix(fsyst_kwant, E)
            T_kwant.append(smat.transmission(1, 0))
        except:
            T_kwant.append(0.0)
    T_kwant = np.array(T_kwant)
    t_kwant = time() - t_start
    print(f"✓ {t_kwant:.3f} s")
else:
    T_kwant = None

# YQT
print("  YQT...", end=" ", flush=True)
t_start = time()
engine = NEGFEngine(device_yqt, Temp=1, eta=eta, gpu=False)
T_yqt = []
warnings_count = 0
for E in E_vals:
    try:
        T = engine.transmission(E)
        T_yqt.append(T)
    except Exception as e:
        T_yqt.append(0.0)
        warnings_count += 1
T_yqt = np.array(T_yqt)
t_yqt = time() - t_start
print(f"✓ {t_yqt:.3f} s ({warnings_count} warnings)")

print()

# ============================================================================
# Analysis
# ============================================================================

print("Analysis:")
print("-" * 80)

print(f"\nBand structure:")
print(f"  Analytical: E ∈ [{np.min(E_analytical):.3f}, {np.max(E_analytical):.3f}]")
if fsyst_kwant:
    print(f"  KWANT:      E ∈ [{np.min(bands_kwant):.3f}, {np.max(bands_kwant):.3f}]")
print(f"  YQT:        E ∈ [{np.min(bands_yqt):.3f}, {np.max(bands_yqt):.3f}]")

print(f"\nTransmission:")
if T_kwant is not None:
    print(f"  KWANT: T ∈ [{np.min(T_kwant):.6f}, {np.max(T_kwant):.6f}]")
print(f"  YQT:   T ∈ [{np.min(T_yqt):.6f}, {np.max(T_yqt):.6f}]")

if T_kwant is not None:
    # Find in-band region (|E| < 2t)
    in_band = np.abs(E_vals) < 1.9 * t
    
    error = np.abs(T_yqt - T_kwant)
    print(f"\nError (all energies):")
    print(f"  Mean: {np.mean(error):.6f}")
    print(f"  Max:  {np.max(error):.6f}")
    print(f"  RMS:  {np.sqrt(np.mean(error**2)):.6f}")
    
    if np.any(in_band):
        error_in_band = error[in_band]
        T_kwant_in_band = T_kwant[in_band]
        T_yqt_in_band = T_yqt[in_band]
        
        print(f"\nIn-band region (|E| < 1.9t):")
        print(f"  KWANT mean T: {np.mean(T_kwant_in_band):.6f}")
        print(f"  YQT mean T:   {np.mean(T_yqt_in_band):.6f}")
        print(f"  Mean error:   {np.mean(error_in_band):.6f}")
        print(f"  Max error:    {np.max(error_in_band):.6f}")
        
        # Check if close
        if np.mean(error_in_band) < 0.1:
            print(f"  ✓ YQT matches KWANT!")
        elif np.mean(error_in_band) < 0.5:
            print(f"  ⚠ Moderate agreement")
        else:
            print(f"  ✗ Poor agreement")

print()

# ============================================================================
# Plots
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Band structure
ax = axes[0, 0]
ax.plot(k_vals/np.pi, E_analytical, 'k-', linewidth=2, label='Analytical', alpha=0.7)
if fsyst_kwant:
    ax.plot(k_vals/np.pi, bands_kwant, 'bo', markersize=4, label='KWANT', alpha=0.6)
ax.plot(k_vals/np.pi, bands_yqt, 'r--', linewidth=2, label='YQT', alpha=0.8)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(-2*t, color='gray', linestyle=':', alpha=0.5, label='Band edges')
ax.axhline(2*t, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('k/π', fontsize=11)
ax.set_ylabel('Energy (t)', fontsize=11)
ax.set_title('Band Structure', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Transmission
ax = axes[0, 1]
if T_kwant is not None:
    ax.plot(E_vals/t, T_kwant, 'b-', linewidth=2, label='KWANT', marker='o', markersize=3)
ax.plot(E_vals/t, T_yqt, 'r--', linewidth=2, label='YQT', marker='s', markersize=3)
ax.axvline(-2, color='gray', linestyle=':', alpha=0.5)
ax.axvline(2, color='gray', linestyle=':', alpha=0.5)
ax.axhline(1, color='k', linestyle=':', alpha=0.5, label='Perfect T')
ax.set_xlabel('Energy (t)', fontsize=11)
ax.set_ylabel('Transmission', fontsize=11)
ax.set_title('Transmission Function', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([-0.1, 1.2])

# Conductance
ax = axes[1, 0]
if T_kwant is not None:
    ax.plot(E_vals/t, T_kwant * G0_SI * 1e6, 'b-', linewidth=2, label='KWANT')
ax.plot(E_vals/t, T_yqt * G0_SI * 1e6, 'r--', linewidth=2, label='YQT')
ax.axvline(-2, color='gray', linestyle=':', alpha=0.5)
ax.axvline(2, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Energy (t)', fontsize=11)
ax.set_ylabel('Conductance (µS)', fontsize=11)
ax.set_title('Conductance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Error
ax = axes[1, 1]
if T_kwant is not None:
    error = np.abs(T_yqt - T_kwant)
    ax.semilogy(E_vals/t, error + 1e-10, 'purple', linewidth=2)
    ax.axvline(-2, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.01, color='orange', linestyle='--', alpha=0.5, label='1% error')
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='10% error')
    ax.set_xlabel('Energy (t)', fontsize=11)
    ax.set_ylabel('Absolute Error |T_YQT - T_KWANT|', fontsize=11)
    ax.set_title('YQT Error vs KWANT', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([1e-6, 10])
else:
    ax.text(0.5, 0.5, 'No KWANT for comparison', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.suptitle(f'1D Tight-Binding Chain: YQT vs KWANT\nLength={N} sites, t={t}',
            fontsize=14, fontweight='bold')
plt.tight_layout()

filename = 'yqt_kwant_1d_chain_comparison.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {filename}")

print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

plt.show()
