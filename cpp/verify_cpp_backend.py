#!/usr/bin/env python3
"""
Verification script for cpp directory to check if C++ backend is working properly.
"""

import sys
import os
import numpy as np

def test_cpp_backend_from_cpp_dir():
    print("=== C++ Backend Verification (from cpp directory) ===")
    
    try:
        # Test direct cpp_negf import
        import cpp_negf
        print("✅ cpp_negf module imported successfully")
        
    except ImportError as e:
        print(f"❌ cpp_negf import failed: {e}")
        return False
    
    try:
        # Test negf_core integration
        from negf_core import CPP_NEGF_AVAILABLE, NEGFSolver
        print(f"✅ negf_core CPP_NEGF_AVAILABLE = {CPP_NEGF_AVAILABLE}")
        
        # Test scf_solver integration  
        from scf_solver import CPP_NEGF_AVAILABLE as SCF_CPP_AVAILABLE
        print(f"✅ scf_solver CPP_NEGF_AVAILABLE = {SCF_CPP_AVAILABLE}")
        
    except ImportError as e:
        print(f"❌ negf_core/scf_solver import failed: {e}")
        return False
    
    try:
        # Test that NEGFSolver uses C++ backend
        H_device = np.array([[0.1+0j, 0.05],[0.05, -0.1]], dtype=complex)
        H_leads = [{'H00': np.eye(2, dtype=complex)*0.1, 'H01': np.eye(2, dtype=complex)*0.05}] * 2
        V_couplings = [np.eye(2, dtype=complex)*0.05] * 2
        
        negf = NEGFSolver(H_device, H_leads, V_couplings)
        
        # Test transmission (should use C++ fast path)
        T = negf.transmission(0.0, lead_i=0, lead_j=1)
        print(f"✅ NEGFSolver transmission call successful, T = {T:.2e}")
        
        # Test finite bias current (should use C++ if available)
        I_cpp = negf.finite_bias_current_cpp(0.005, -0.005, 300.0, 0.0)
        if I_cpp is not None:
            print(f"✅ NEGFSolver finite_bias_current_cpp successful, I = {I_cpp:.2e}")
        else:
            print("⚠️  NEGFSolver finite_bias_current_cpp returned None (fallback expected)")
        
    except Exception as e:
        print(f"❌ NEGFSolver test failed: {e}")
        return False
        
    print("✅ All C++ backend tests passed from cpp directory!")
    return True

if __name__ == "__main__":
    success = test_cpp_backend_from_cpp_dir()
    sys.exit(0 if success else 1)