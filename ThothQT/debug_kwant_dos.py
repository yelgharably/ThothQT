"""
Debug KWANT DOS calculation to understand the discrepancy
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def debug_kwant_dos():
    """Debug KWANT DOS calculation"""
    print("Debugging KWANT DOS Calculation")
    print("=" * 40)
    
    # Create KWANT system
    n_sites = 4
    t = 1.0
    
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
    
    # ThothQT system for comparison
    device_tqt = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
    
    # Test at a few energies
    test_energies = [0.0, 0.5, 1.0]
    
    print(f"{'E':>6} {'ThothQT':>12} {'KWANT LDOS':>12} {'KWANT Sum':>12}")
    print("-" * 48)
    
    for E in test_energies:
        # ThothQT DOS
        dos_tqt = engine_tqt.density_of_states(E)
        
        # KWANT LDOS
        try:
            ldos_kwant = kwant.ldos(finalized_syst, E)
            ldos_sum = np.sum(ldos_kwant)
            print(f"{E:6.1f} {dos_tqt:12.6f} {ldos_kwant[0]:12.6f} {ldos_sum:12.6f}")
            print(f"       KWANT LDOS array: {ldos_kwant}")
        except Exception as e:
            print(f"{E:6.1f} {dos_tqt:12.6f} ERROR: {e}")
    
    # Let's also check what happens with the Green's function approach
    print(f"\nDirect Green's function check at E=0:")
    try:
        # Get Green's function matrix elements from KWANT
        # This is more complex but let's see if we can extract it
        
        # Alternative: use KWANT's wave function approach
        # Let's check the actual LDOS calculation in more detail
        E = 0.0
        ldos_kwant = kwant.ldos(finalized_syst, E)
        
        print(f"KWANT LDOS shape: {ldos_kwant.shape}")
        print(f"KWANT LDOS values: {ldos_kwant}")
        print(f"KWANT LDOS sum: {np.sum(ldos_kwant)}")
        
        # Check what sites we have
        device_sites = list(finalized_syst.sites)
        print(f"Device sites: {len(device_sites)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_kwant_dos()