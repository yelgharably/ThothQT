"""
Debug DOS and Band Structure implementations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def debug_dos_bands():
    """Debug DOS and band structure calculations"""
    print("Debugging DOS and Band Structure")
    print("=" * 40)
    
    # Simple system
    n_sites = 4
    t = 1.0
    
    # ThothQT system
    device_tqt = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
    
    print(f"ThothQT system:")
    print(f"Device H:\n{device_tqt.H.toarray()}")
    
    # Test DOS at a few points
    test_energies = [0.0, 0.5, 1.0]
    print(f"\nDOS comparison:")
    print(f"{'E':>6} {'ThothQT':>12}")
    print("-" * 20)
    
    for E in test_energies:
        dos_tqt = engine_tqt.density_of_states(E)
        print(f"{E:6.1f} {dos_tqt:12.6f}")
    
    # Test band structure
    k_points = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    bands_tqt = engine_tqt.band_structure(k_points)
    
    print(f"\nBand structure comparison:")
    print(f"{'k':>8} {'ThothQT':>12} {'Analytical':>12}")
    print("-" * 34)
    
    for k, E_tqt in zip(k_points, bands_tqt):
        # Analytical 1D tight-binding: E(k) = 2t*cos(k)
        E_analytical = 2 * t * np.cos(k)
        print(f"{k:8.3f} {E_tqt:12.6f} {E_analytical:12.6f}")
    
    # Create a simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # DOS plot
    E_range = np.linspace(-2.5, 2.5, 51)
    dos_values = [engine_tqt.density_of_states(E) for E in E_range]
    
    ax1.plot(E_range, dos_values, 'b-', linewidth=2)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('DOS')
    ax1.set_title('ThothQT DOS')
    ax1.grid(True, alpha=0.3)
    
    # Band structure plot
    k_range = np.linspace(-np.pi, np.pi, 50)
    bands_tqt_full = engine_tqt.band_structure(k_range)
    bands_analytical = 2 * t * np.cos(k_range)
    
    ax2.plot(k_range, bands_tqt_full, 'b-', linewidth=2, label='ThothQT')
    ax2.plot(k_range, bands_analytical, 'r--', linewidth=2, label='Analytical')
    ax2.set_xlabel('k (1/a)')
    ax2.set_ylabel('Energy (eV)')
    ax2.set_title('Band Structure')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_dos_bands.png', dpi=150, bbox_inches='tight')
    print("\nDebug plot saved: debug_dos_bands.png")
    
    try:
        plt.show()
    except:
        print("Plot display not available")

if __name__ == "__main__":
    debug_dos_bands()