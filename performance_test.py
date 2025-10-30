#!/usr/bin/env python3
"""
Performance test to check if C++ backend is actually providing speedup
"""

import time
import numpy as np
import sys
import os

# Add cpp directory
sys.path.insert(0, 'cpp')

def performance_test():
    print('=== Performance Test ===')
    
    # Import modules
    from negf_core import NEGFSolver, CPP_NEGF_AVAILABLE
    print(f'CPP_NEGF_AVAILABLE: {CPP_NEGF_AVAILABLE}')
    
    # Create test system (medium size)
    n_device = 50
    n_lead = 10
    
    H_device = np.random.random((n_device, n_device)) + 1j*np.random.random((n_device, n_device))
    H_device = (H_device + H_device.T.conj()) / 2  # Make Hermitian
    
    H_leads = []
    V_couplings = []
    for i in range(2):
        H00 = np.random.random((n_lead, n_lead)) + 1j*np.random.random((n_lead, n_lead))
        H00 = (H00 + H00.T.conj()) / 2
        H01 = np.random.random((n_lead, n_lead)) + 1j*np.random.random((n_lead, n_lead))
        H_leads.append({'H00': H00, 'H01': H01})
        
        V_coupling = np.random.random((n_lead, n_device)) + 1j*np.random.random((n_lead, n_device))
        V_couplings.append(V_coupling)
    
    negf = NEGFSolver(H_device, H_leads, V_couplings)
    
    # Test 1: Single transmission calculation
    print('\n--- Single Transmission Test ---')
    start = time.time()
    T1 = negf.transmission(0.0, 0, 1)
    end = time.time()
    print(f'Transmission: {(end-start)*1000:.1f} ms, T = {T1:.2e}')
    
    # Test 2: Multiple transmission calculations (sweep-like)
    print('\n--- Multiple Transmission Test ---')
    energies = np.linspace(-0.1, 0.1, 21)
    start = time.time()
    transmissions = []
    for E in energies:
        T = negf.transmission(E, 0, 1)
        transmissions.append(T)
    end = time.time()
    print(f'21 transmissions: {(end-start)*1000:.1f} ms total, {(end-start)*1000/21:.1f} ms/calc')
    
    # Test 3: Finite bias current with C++
    print('\n--- Finite Bias Current Test ---')
    start = time.time()
    I_cpp = negf.finite_bias_current_cpp(0.005, -0.005, 300.0, 0.0, NE=51)
    end = time.time()
    if I_cpp is not None:
        print(f'C++ finite bias current: {(end-start)*1000:.1f} ms, I = {I_cpp:.2e}')
    else:
        print('C++ finite bias current returned None (fallback)')
        
    # Test 4: Direct C++ calls
    print('\n--- Direct C++ Function Test ---')
    try:
        import cpp_negf
        
        # Test surface GF (should be fast)
        start = time.time()
        gs = cpp_negf.surface_gf(H_leads[0]['H00'], H_leads[0]['H01'], 0.0, 1e-6)
        end = time.time()
        print(f'C++ surface_gf: {(end-start)*1000:.1f} ms')
        
        # Test transmission from leads (should be fast)  
        start = time.time()
        T_cpp = cpp_negf.transmission_from_leads(
            H_device, 
            H_leads[0]['H00'], H_leads[0]['H01'], V_couplings[0],
            H_leads[1]['H00'], H_leads[1]['H01'], V_couplings[1],
            0.0, 1e-6
        )
        end = time.time()
        print(f'C++ transmission_from_leads: {(end-start)*1000:.1f} ms, T = {T_cpp:.2e}')
        
        # Test transmission sweep (should be much faster)
        start = time.time()
        T_sweep = cpp_negf.transmission_sweep_from_leads(
            H_device,
            H_leads[0]['H00'], H_leads[0]['H01'], V_couplings[0],
            H_leads[1]['H00'], H_leads[1]['H01'], V_couplings[1],
            energies, 1e-6
        )
        end = time.time()
        print(f'C++ transmission sweep (21 points): {(end-start)*1000:.1f} ms, {(end-start)*1000/21:.1f} ms/calc')
        
    except Exception as e:
        print(f'Direct C++ test failed: {e}')

if __name__ == '__main__':
    performance_test()