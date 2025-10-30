"""
Test script to validate NEGF implementation against Kwant S-matrix results.

This script creates a simple test system and compares the transmission
calculated using NEGF formalism vs. Kwant's S-matrix approach.
"""

import numpy as np
import sys
import os

# Add the negf-hyb directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'negf-hyb'))

try:
    import kwant
    from negf_core import NEGFSolver, extract_kwant_matrices
    
    def create_simple_chain(length=10, t=1.0):
        """Create a simple 1D chain for testing."""
        lat = kwant.lattice.chain(norbs=1)
        sys = kwant.Builder()
        
        # Central region
        for i in range(length):
            sys[lat(i)] = 0.0
        
        for i in range(length-1):
            sys[lat(i), lat(i+1)] = -t
        
        # Leads
        lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
        lead[lat(0)] = 0.0
        lead[lat(1), lat(0)] = -t
        
        sys.attach_lead(lead)
        sys.attach_lead(lead.reversed())
        
        return sys.finalized()
    
    def test_negf_vs_kwant():
        """Test NEGF implementation against Kwant S-matrix."""
        print("Testing NEGF implementation...")
        
        # Create test system
        fsys = create_simple_chain(length=5, t=1.0)
        
        # Test parameters
        energies = np.linspace(-2.0, 2.0, 11)
        params = {}
        
        print(f"{'Energy':>8} {'T_Kwant':>10} {'T_NEGF':>10} {'Difference':>12}")
        print("-" * 50)
        
        for E in energies:
            # Kwant S-matrix result
            try:
                smat = kwant.smatrix(fsys, energy=E, params=params)
                T_kwant = smat.transmission(0, 1)
            except Exception as e:
                print(f"Kwant failed at E={E:.2f}: {e}")
                continue
            
            # NEGF result
            try:
                H_device, H_leads, V_couplings = extract_kwant_matrices(fsys, E, params)
                negf = NEGFSolver(H_device, H_leads, V_couplings, eta=1e-6)
                T_negf = negf.transmission(E, lead_i=0, lead_j=1)
            except Exception as e:
                print(f"NEGF failed at E={E:.2f}: {e}")
                T_negf = 0.0
            
            diff = abs(T_kwant - T_negf)
            print(f"{E:8.2f} {T_kwant:10.6f} {T_negf:10.6f} {diff:12.8f}")
        
        print("\nTest completed!")
    
    if __name__ == "__main__":
        test_negf_vs_kwant()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure Kwant and the NEGF module are available.")