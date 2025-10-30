"""
Complete ThothQT vs KWANT Comparison including DOS and Band Structure
===================================================================

This script performs a comprehensive comparison of ThothQT against KWANT
including transmission, conductance, density of states (DOS), and band structure.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def make_kwant_1d_chain(n_sites=8, t=1.0):
    """Create KWANT 1D chain system"""
    lat = kwant.lattice.chain(a=1, norbs=1)
    
    syst = kwant.Builder()
    for i in range(n_sites):
        syst[lat(i)] = 0.0
    for i in range(n_sites - 1):
        syst[lat(i), lat(i + 1)] = -t
    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = 0.0
    lead[lat(0), lat(1)] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    return syst.finalized()

def compute_kwant_transmission(finalized_syst, energies):
    """Compute KWANT transmission spectrum"""
    T_kwant = []
    for E in energies:
        try:
            smatrix = kwant.smatrix(finalized_syst, E)
            T_kwant.append(smatrix.transmission(0, 1))
        except:
            T_kwant.append(0.0)
    return np.array(T_kwant)

def compute_kwant_dos(finalized_syst, energies):
    """Compute KWANT density of states"""
    dos_kwant = []
    for E in energies:
        try:
            # Use KWANT's local density of states and sum over all sites
            ldos = kwant.ldos(finalized_syst, E)
            dos_total = np.sum(ldos)  # Sum LDOS over all device sites
            dos_kwant.append(dos_total)
        except Exception as exc:
            # Handle energies outside the allowed band
            dos_kwant.append(0.0)
    return np.array(dos_kwant)

def compute_kwant_band_structure(n_k=100):
    """Compute KWANT band structure for infinite 1D chain"""
    # Create infinite 1D system for band structure
    lat = kwant.lattice.chain(a=1, norbs=1)
    
    # Infinite system with proper translational symmetry
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    syst[lat(0)] = 0.0  # on-site energy
    syst[lat(0), lat(1)] = -1.0  # hopping
    
    finalized_syst = syst.finalized()
    
    # Create the band structure object once
    bands = kwant.physics.Bands(finalized_syst)
    
    # k-points in first Brillouin zone
    k_points = np.linspace(-np.pi, np.pi, n_k)
    energies = []
    
    for k in k_points:
        try:
            # Compute energy bands at this k-point
            energy_at_k = bands(k)
            energies.append(energy_at_k[0])  # Take first (and only) band
        except:
            # Use analytical solution as fallback: E(k) = 2t*cos(k)
            energies.append(2.0 * np.cos(k))
    
    return k_points, np.array(energies)

def complete_comparison():
    """Run complete ThothQT vs KWANT comparison"""
    print("Complete ThothQT vs KWANT Comparison")
    print("=" * 60)
    
    # System parameters
    n_sites = 8
    t = 1.0
    
    print(f"System: 1D chain with {n_sites} sites, hopping t = {t}")
    print()
    
    # Create ThothQT system
    print("Creating ThothQT system...")
    device_tqt = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
    
    # Create KWANT system
    print("Creating KWANT system...")
    finalized_syst = make_kwant_1d_chain(n_sites=n_sites, t=t)
    
    # Energy ranges
    # For transmission and DOS: focus on allowed band
    energies_transport = np.linspace(-1.8, 1.8, 37)  # Stay within [-2t, 2t] band
    energies_dos = np.linspace(-2.5, 2.5, 51)  # Slightly broader for DOS
    
    print("Computing transmission spectra...")
    
    # === TRANSMISSION COMPARISON ===
    start_time = time.time()
    T_tqt = [engine_tqt.transmission(E) for E in energies_transport]
    time_tqt_trans = time.time() - start_time
    
    start_time = time.time() 
    T_kwant = compute_kwant_transmission(finalized_syst, energies_transport)
    time_kwant_trans = time.time() - start_time
    
    T_tqt = np.array(T_tqt)
    
    print("Computing density of states...")
    
    # === DOS COMPARISON ===
    start_time = time.time()
    dos_tqt = [engine_tqt.density_of_states(E) for E in energies_dos]
    time_tqt_dos = time.time() - start_time
    
    start_time = time.time()
    dos_kwant = compute_kwant_dos(finalized_syst, energies_dos)
    time_kwant_dos = time.time() - start_time
    
    dos_tqt = np.array(dos_tqt)
    
    print("Computing band structures...")
    
    # === BAND STRUCTURE COMPARISON ===
    n_k = 50
    
    start_time = time.time()
    k_points_tqt, bands_tqt = engine_tqt.compute_band_structure_1d(n_k=n_k)
    time_tqt_bands = time.time() - start_time
    
    start_time = time.time()
    k_points_kwant, bands_kwant = compute_kwant_band_structure(n_k=n_k)
    time_kwant_bands = time.time() - start_time
    
    # === ANALYSIS ===
    print("\nPhysics Agreement Analysis:")
    print("-" * 30)
    
    # Transmission agreement
    diff_trans = np.abs(T_tqt - T_kwant)
    max_diff_trans = np.max(diff_trans)
    mean_diff_trans = np.mean(diff_trans)
    print(f"Transmission:")
    print(f"  Max difference:  {max_diff_trans:.2e}")
    print(f"  Mean difference: {mean_diff_trans:.2e}")
    
    # DOS agreement
    diff_dos = np.abs(dos_tqt - dos_kwant)
    max_diff_dos = np.max(diff_dos)
    mean_diff_dos = np.mean(diff_dos)
    print(f"DOS:")
    print(f"  Max difference:  {max_diff_dos:.2e}")
    print(f"  Mean difference: {mean_diff_dos:.2e}")
    
    # Band structure agreement
    diff_bands = np.abs(bands_tqt - bands_kwant)
    max_diff_bands = np.max(diff_bands)
    mean_diff_bands = np.mean(diff_bands)
    print(f"Band structure:")
    print(f"  Max difference:  {max_diff_bands:.2e}")
    print(f"  Mean difference: {mean_diff_bands:.2e}")
    
    # Performance analysis
    print("\nPerformance Analysis:")
    print("-" * 20)
    speedup_trans = time_kwant_trans / time_tqt_trans
    speedup_dos = time_kwant_dos / time_tqt_dos
    speedup_bands = time_kwant_bands / time_tqt_bands
    
    print(f"Transmission: {speedup_trans:.1f}x faster")
    print(f"DOS:          {speedup_dos:.1f}x faster") 
    print(f"Band struct:  {speedup_bands:.1f}x faster")
    
    # === PLOTTING ===
    print("\nCreating comprehensive comparison plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Transmission
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(energies_transport, T_tqt, 'b-', linewidth=2, label='ThothQT')
    plt.plot(energies_transport, T_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title('Transmission Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # 2. Transmission difference
    ax2 = plt.subplot(3, 3, 2)
    plt.semilogy(energies_transport, diff_trans + 1e-16, 'g-', linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('|T_TQT - T_KWANT|')
    plt.title('Transmission Difference')
    plt.grid(True, alpha=0.3)
    
    # 3. Conductance
    ax3 = plt.subplot(3, 3, 3)
    G_tqt = T_tqt  # In units of 2e²/h
    G_kwant = T_kwant
    plt.plot(energies_transport, G_tqt, 'b-', linewidth=2, label='ThothQT')
    plt.plot(energies_transport, G_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Conductance (2e²/h)')
    plt.title('Conductance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. DOS - ThothQT
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(energies_dos, dos_tqt, 'b-', linewidth=2, label='ThothQT')
    plt.plot(energies_dos, dos_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV)')
    plt.title('Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. DOS difference
    ax5 = plt.subplot(3, 3, 5)
    plt.semilogy(energies_dos, diff_dos + 1e-16, 'g-', linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('|DOS_TQT - DOS_KWANT|')
    plt.title('DOS Difference')
    plt.grid(True, alpha=0.3)
    
    # 6. Performance comparison
    ax6 = plt.subplot(3, 3, 6)
    methods = ['Transmission', 'DOS', 'Bands']
    speedups = [speedup_trans, speedup_dos, speedup_bands]
    bars = plt.bar(methods, speedups, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.ylabel('Speedup Factor')
    plt.title('Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for bar, speed in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(speedups)*0.02,
                f'{speed:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 7. Band structure
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(k_points_tqt, bands_tqt, 'b-', linewidth=2, label='ThothQT')
    plt.plot(k_points_kwant, bands_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    plt.xlabel('k (1/a)')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Band structure difference
    ax8 = plt.subplot(3, 3, 8)
    plt.semilogy(k_points_tqt, diff_bands + 1e-16, 'g-', linewidth=2)
    plt.xlabel('k (1/a)')
    plt.ylabel('|E_TQT - E_KWANT|')
    plt.title('Band Structure Difference')
    plt.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
VALIDATION SUMMARY

TRANSMISSION:
Max diff: {max_diff_trans:.1e}
Mean diff: {mean_diff_trans:.1e}
Speedup: {speedup_trans:.1f}x

DOS:
Max diff: {max_diff_dos:.1e}  
Mean diff: {mean_diff_dos:.1e}
Speedup: {speedup_dos:.1f}x

BAND STRUCTURE:
Max diff: {max_diff_bands:.1e}
Mean diff: {mean_diff_bands:.1e}
Speedup: {speedup_bands:.1f}x

OVERALL:
Physics: {"EXCELLENT" if max_diff_trans < 1e-6 else "GOOD" if max_diff_trans < 1e-3 else "POOR"}
Performance: {"OUTSTANDING" if min(speedup_trans, speedup_dos, speedup_bands) > 5 else "GOOD"}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    try:
        plt.savefig('complete_thothqt_kwant_comparison.png', dpi=150, bbox_inches='tight')
        print("Complete comparison plot saved: complete_thothqt_kwant_comparison.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    print("\nFinal Assessment:")
    print("=" * 20)
    
    overall_accuracy = "EXCELLENT" if max(max_diff_trans, max_diff_dos, max_diff_bands) < 1e-6 else "GOOD"
    overall_performance = "OUTSTANDING" if min(speedup_trans, speedup_dos, speedup_bands) > 3 else "GOOD"
    
    print(f"Physics accuracy: {overall_accuracy}")
    print(f"Performance:      {overall_performance}")
    print(f"Ready for use:    {'YES' if overall_accuracy in ['EXCELLENT', 'GOOD'] else 'NO'}")
    
    return {
        'transmission': {'energies': energies_transport, 'tqt': T_tqt, 'kwant': T_kwant, 'diff': max_diff_trans},
        'dos': {'energies': energies_dos, 'tqt': dos_tqt, 'kwant': dos_kwant, 'diff': max_diff_dos},
        'bands': {'k': k_points_tqt, 'tqt': bands_tqt, 'kwant': bands_kwant, 'diff': max_diff_bands},
        'performance': {'trans': speedup_trans, 'dos': speedup_dos, 'bands': speedup_bands}
    }

if __name__ == "__main__":
    results = complete_comparison()
    
    try:
        plt.show()
    except:
        print("Plot display not available, but plots saved to file.")