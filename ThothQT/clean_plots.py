"""
Clean ThothQT vs KWANT Comparison Plots
=======================================

Simple, clean plotting script to visualize the comparison results
without memory issues or complex layouts.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def run_simple_comparison():
    """Run a simple comparison and create clean plots"""
    print("Running Simple ThothQT vs KWANT Comparison")
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
    
    # 2. Transmission comparison over energy range
    energies = np.linspace(-2, 2, 41)
    
    print("Computing transmission spectra...")
    
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
    
    # 3. Graphene nanoribbon test
    print("Computing graphene nanoribbon...")
    builder = tqt.GrapheneBuilder(a=1.42, t=2.7)
    device_gr = builder.zigzag_ribbon(width=3, length=4)
    engine_gr = tqt.NEGFEngine(device_gr, Temp=300)
    
    energies_gr = np.linspace(-4, 4, 21)
    T_graphene = [engine_gr.transmission(E) for E in energies_gr]
    
    return {
        'energies': energies,
        'T_tqt': np.array(T_tqt),
        'T_kwant': np.array(T_kwant),
        'time_tqt': time_tqt,
        'time_kwant': time_kwant,
        'energies_gr': energies_gr,
        'T_graphene': np.array(T_graphene),
        'n_sites': n_sites
    }

def create_clean_plots(results):
    """Create clean, simple plots"""
    print("\nCreating comparison plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Transmission comparison
    ax1.plot(results['energies'], results['T_tqt'], 'b-', linewidth=2, label='ThothQT')
    ax1.plot(results['energies'], results['T_kwant'], 'r--', linewidth=2, label='KWANT')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission')
    ax1.set_title(f'1D Chain Transmission (N={results["n_sites"]} sites)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)  # Show physical transmission range clearly
    
    # 2. Difference plot
    diff = np.abs(results['T_tqt'] - results['T_kwant'])
    ax2.semilogy(results['energies'], diff + 1e-16, 'g-', linewidth=2)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('|T_ThothQT - T_KWANT|')
    ax2.set_title('Transmission Difference')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-16, 1e-10)
    
    # Add text box with statistics
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    ax2.text(0.05, 0.95, f'Max: {max_diff:.1e}\nMean: {mean_diff:.1e}', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    # 3. Performance comparison
    methods = ['ThothQT', 'KWANT']
    times = [results['time_tqt'], results['time_kwant']]
    throughput = [len(results['energies'])/t for t in times]
    
    bars = ax3.bar(methods, throughput, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Calculations per Second')
    ax3.set_title('Performance Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, tp in zip(bars, throughput):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(throughput)*0.02,
                f'{tp:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add speedup text
    speedup = results['time_kwant'] / results['time_tqt']
    ax3.text(0.5, max(throughput)*0.7, f'{speedup:.1f}x faster',
             ha='center', va='center', transform=ax3.transData,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # 4. Graphene nanoribbon
    ax4.plot(results['energies_gr'], results['T_graphene'], 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Energy (eV)')
    ax4.set_ylabel('Transmission')
    ax4.set_title('Graphene Zigzag Nanoribbon')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(results['T_graphene'])*1.1)
    
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig('thothqt_kwant_clean_comparison.png', dpi=150, bbox_inches='tight')
        print("Clean comparison plots saved: thothqt_kwant_clean_comparison.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    return fig

def print_summary(results):
    """Print clean summary"""
    print("\n" + "="*60)
    print("THOTHQT VS KWANT COMPARISON SUMMARY")
    print("="*60)
    
    # Physics accuracy
    diff = np.abs(results['T_tqt'] - results['T_kwant'])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nPHYSICS ACCURACY:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Mean difference:    {mean_diff:.2e}")
    
    if max_diff < 1e-10:
        accuracy = "PERFECT (machine precision)"
    elif max_diff < 1e-6:
        accuracy = "EXCELLENT"
    else:
        accuracy = "POOR"
    
    print(f"  Agreement level:    {accuracy}")
    
    # Performance
    speedup = results['time_kwant'] / results['time_tqt']
    tqt_throughput = len(results['energies']) / results['time_tqt']
    kwant_throughput = len(results['energies']) / results['time_kwant']
    
    print(f"\nPERFORMANCE:")
    print(f"  ThothQT time:       {results['time_tqt']:.3f} seconds")
    print(f"  KWANT time:         {results['time_kwant']:.3f} seconds")
    print(f"  Speedup factor:     {speedup:.1f}x faster")
    print(f"  ThothQT throughput: {tqt_throughput:.0f} calc/s")
    print(f"  KWANT throughput:   {kwant_throughput:.0f} calc/s")
    
    # System info
    print(f"\nSYSTEM DETAILS:")
    print(f"  1D chain length:    {results['n_sites']} sites")
    print(f"  Energy points:      {len(results['energies'])}")
    print(f"  Graphene system:    {len(results['T_graphene'])} energy points")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    if accuracy == "PERFECT (machine precision)" and speedup > 5:
        grade = "OUTSTANDING SUCCESS"
    elif accuracy in ["PERFECT (machine precision)", "EXCELLENT"] and speedup > 2:
        grade = "EXCELLENT SUCCESS"
    else:
        grade = "NEEDS IMPROVEMENT"
    
    print(f"  ThothQT vs KWANT:   {grade}")
    print(f"  Ready for production use: {'YES' if grade != 'NEEDS IMPROVEMENT' else 'NO'}")

def main():
    """Main function"""
    # Run comparison
    results = run_simple_comparison()
    
    # Create plots
    fig = create_clean_plots(results)
    
    # Print summary
    print_summary(results)
    
    # Show plots
    try:
        plt.show()
    except:
        print("\nPlot display not available, but plots saved to file.")
    
    return results

if __name__ == "__main__":
    results = main()