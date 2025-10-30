"""
Debug ThothQT vs KWANT transmission calculation
"""

import sys
import os
import numpy as np

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def debug_transmission():
    """Debug transmission calculation at a few key points"""
    print("ThothQT vs KWANT Debug Comparison")
    print("=" * 40)
    
    # 1. Create identical systems
    n_sites = 4  # Start small
    t = 1.0
    
    # ThothQT system
    device_tqt = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
    
    print(f"ThothQT system:")
    print(f"Device H shape: {device_tqt.H.shape}")
    print(f"Device H:\n{device_tqt.H.toarray()}")
    print(f"Left lead H00: {device_tqt.left.H00}")
    print(f"Left lead H01: {device_tqt.left.H01}")
    
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
    
    print(f"\nKWANT system created successfully")
    
    # 2. Test at specific energies
    test_energies = [0.0, 0.5, 1.0, 2.0]
    
    print(f"\nTransmission comparison:")
    print(f"{'E':>6} {'T_ThothQT':>12} {'T_KWANT':>12} {'Difference':>12}")
    print("-" * 50)
    
    for E in test_energies:
        # ThothQT transmission
        T_tqt = engine_tqt.transmission(E)
        
        # KWANT transmission
        try:
            smatrix = kwant.smatrix(finalized_syst, E)
            T_kwant = smatrix.transmission(0, 1)
        except:
            T_kwant = 0.0
        
        diff = abs(T_tqt - T_kwant)
        print(f"{E:6.1f} {T_tqt:12.6f} {T_kwant:12.6f} {diff:12.6f}")
        
        # Debug details for E=0
        if abs(E) < 1e-10:
            print(f"\nDetailed debug at E=0:")
            
            # Check self-energies
            SigmaL = device_tqt.left.sigma(E, engine_tqt.eta)
            SigmaR = device_tqt.right.sigma(E, engine_tqt.eta)
            print(f"SigmaL: {SigmaL}")
            print(f"SigmaR: {SigmaR}")
            
            # Check broadening matrices
            N = device_tqt.H.shape[0]
            SigmaL_full = np.zeros((N, N), dtype=complex)
            SigmaR_full = np.zeros((N, N), dtype=complex)
            
            # Handle sparse or scalar self-energies
            if hasattr(SigmaL, 'toarray'):
                SigmaL_val = SigmaL.toarray()[0, 0] if SigmaL.shape == (N, N) else SigmaL.data[0] if SigmaL.nnz > 0 else 0
            else:
                SigmaL_val = SigmaL
                
            if hasattr(SigmaR, 'toarray'):
                SigmaR_val = SigmaR.toarray()[-1, -1] if SigmaR.shape == (N, N) else SigmaR.data[0] if SigmaR.nnz > 0 else 0
            else:
                SigmaR_val = SigmaR
                
            SigmaL_full[0, 0] = SigmaL_val
            SigmaR_full[-1, -1] = SigmaR_val
            
            GammaL = 1j * (SigmaL_full - SigmaL_full.T.conj())
            GammaR = 1j * (SigmaR_full - SigmaR_full.T.conj())
            
            print(f"GammaL:\n{GammaL}")
            print(f"GammaR:\n{GammaR}")
            
            # Check Green's function matrix
            import scipy.sparse as sp
            I = sp.eye(N, dtype=complex, format='csr')
            A = E * I - device_tqt.H - sp.csr_matrix(SigmaL_full) - sp.csr_matrix(SigmaR_full)
            print(f"System matrix A:\n{A.toarray()}")
            
            try:
                G = np.linalg.inv(A.toarray())
                print(f"Green's function G:\n{G}")
                
                # Manual Fisher-Lee calculation
                manual_T = np.trace(GammaL @ G @ GammaR @ G.T.conj())
                print(f"Manual T calculation: {manual_T}")
                print(f"Real part: {np.real(manual_T)}")
                
            except Exception as e:
                print(f"Green's function calculation failed: {e}")
    
if __name__ == "__main__":
    debug_transmission()