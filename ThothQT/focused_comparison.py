"""
Focused ThothQT vs KWANT comparison within the allowed energy band
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def focused_comparison():
    """Compare ThothQT vs KWANT within the physical energy range"""
    print("Focused ThothQT vs KWANT Comparison")
    print("=" * 50)
    
    # 1. Simple 1D chain test
    n_sites = 8
    t = 1.0
    
    # ThothQT system
    device_tqt = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
    
    # KWANT system
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
    finalized_syst = syst.finalized()
    
    # 2. Focus on the allowed energy band [-2t, 2t] = [-2, 2]
    # But stay away from band edges where numerical issues occur
    energies = np.linspace(-1.8, 1.8, 37)  # Stay within band, avoid edges
    
    print("Computing transmission spectra within allowed band...")
    print(f"Energy range: [{energies[0]:.1f}, {energies[-1]:.1f}] eV")
    
    # ThothQT transmissions
    import time
    start = time.time()
    T_tqt = [engine_tqt.transmission(E) for E in energies]
    time_tqt = time.time() - start
    
    # KWANT transmissions
    start = time.time()
    T_kwant = []
    for E in energies:
        try:
            smatrix = kwant.smatrix(finalized_syst, E)
            T_kwant.append(smatrix.transmission(0, 1))
        except:
            T_kwant.append(0.0)
    time_kwant = time.time() - start
    
    # Convert to arrays
    T_tqt = np.array(T_tqt)
    T_kwant = np.array(T_kwant)
    
    # 3. Analyze agreement
    diff = np.abs(T_tqt - T_kwant)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nPhysics Agreement (within allowed band):")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Mean difference:    {mean_diff:.2e}")
    print(f"  RMS difference:     {np.sqrt(np.mean(diff**2)):.2e}")
    
    # Find the worst agreement points
    worst_idx = np.argmax(diff)
    print(f"  Worst agreement at E={energies[worst_idx]:.3f}: T_TQT={T_tqt[worst_idx]:.6f}, T_KWANT={T_kwant[worst_idx]:.6f}")
    
    # Check for perfect agreement points
    perfect_points = np.sum(diff < 1e-10)
    print(f"  Points with perfect agreement (< 1e-10): {perfect_points}/{len(energies)}")
    
    # 4. Performance comparison  
    speedup = time_kwant / time_tqt
    tqt_throughput = len(energies) / time_tqt
    kwant_throughput = len(energies) / time_kwant
    
    print(f"\nPerformance (within allowed band):")
    print(f"  ThothQT time:       {time_tqt:.3f} seconds")
    print(f"  KWANT time:         {time_kwant:.3f} seconds")
    print(f"  Speedup factor:     {speedup:.1f}x faster")
    print(f"  ThothQT throughput: {tqt_throughput:.0f} calc/s")
    print(f"  KWANT throughput:   {kwant_throughput:.0f} calc/s")
    
    # 5. Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Transmission comparison
    ax1.plot(energies, T_tqt, 'b-', linewidth=2, label='ThothQT')
    ax1.plot(energies, T_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission')
    ax1.set_title(f'1D Chain Transmission (N={n_sites} sites, within allowed band)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)  # Show physical transmission range
    
    # Difference plot
    ax2.semilogy(energies, diff + 1e-16, 'g-', linewidth=2)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('|T_ThothQT - T_KWANT|')
    ax2.set_title('Transmission Difference (log scale)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-16, max(1e-10, max_diff * 2))
    
    # Add statistics text box
    ax2.text(0.05, 0.95, f'Max: {max_diff:.1e}\\nMean: {mean_diff:.1e}\\nPerfect points: {perfect_points}/{len(energies)}', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    try:
        plt.savefig('focused_thothqt_kwant_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nFocused comparison plot saved: focused_thothqt_kwant_comparison.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # 6. Overall assessment
    print(f"\nOVERALL ASSESSMENT (FOCUSED):")
    if max_diff < 1e-12:
        accuracy = "PERFECT (machine precision)"
        grade = "OUTSTANDING"
    elif max_diff < 1e-6:
        accuracy = "EXCELLENT"
        grade = "EXCELLENT" if speedup > 5 else "GOOD"
    elif max_diff < 1e-3:
        accuracy = "GOOD"  
        grade = "ACCEPTABLE" if speedup > 2 else "POOR"
    else:
        accuracy = "POOR"
        grade = "NEEDS IMPROVEMENT"
    
    print(f"  Physics accuracy:   {accuracy}")
    print(f"  Performance grade:  {grade}")
    print(f"  Ready for use:      {'YES' if grade in ['OUTSTANDING', 'EXCELLENT'] else 'MAYBE' if grade == 'ACCEPTABLE' else 'NO'}")
    
    # Show a few sample calculations
    print(f"\nSample transmission values:")
    print(f"{'Energy':>8} {'ThothQT':>10} {'KWANT':>10} {'|Diff|':>10}")
    print("-" * 42)
    for i in [0, len(energies)//4, len(energies)//2, 3*len(energies)//4, -1]:
        E, T_t, T_k = energies[i], T_tqt[i], T_kwant[i]
        print(f"{E:8.2f} {T_t:10.6f} {T_k:10.6f} {abs(T_t-T_k):10.1e}")
    
    return {
        'energies': energies,
        'T_tqt': T_tqt,
        'T_kwant': T_kwant,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'speedup': speedup,
        'accuracy': accuracy,
        'grade': grade
    }

if __name__ == "__main__":
    results = focused_comparison()
    
    try:
        plt.show()
    except:
        print("Plot display not available, but plot saved to file.")