#!/usr/bin/env python3
"""
Verification script to check if C++ backend is working properly.
"""

import sys
import os
import numpy as np

# Add cpp directory to path (same as main script)
_repo_dir = os.path.dirname(__file__)
_cpp_dir = os.path.join(_repo_dir, 'cpp')
if os.path.isdir(_cpp_dir):
    sys.path.insert(0, _cpp_dir)

def test_cpp_backend():
    print("=== C++ Backend Verification ===")
    
    try:
        # Test direct cpp_negf import
        import cpp_negf
        print("✅ cpp_negf module imported successfully")
        
        # Test the functions exist
        funcs_to_check = [
            'transmission_dense', 'finite_bias_current',
            'surface_gf', 'self_energy_from_lead', 
            'transmission_from_leads', 'transmission_sweep_from_leads',
            'finite_bias_current_from_leads'
        ]
        
        for func_name in funcs_to_check:
            if hasattr(cpp_negf, func_name):
                print(f"✅ cpp_negf.{func_name} available")
            else:
                print(f"❌ cpp_negf.{func_name} missing")
                
    except ImportError as e:
        print(f"❌ cpp_negf import failed: {e}")
        return False
    
    try:
        # Test negf_core integration
        from negf_core import CPP_NEGF_AVAILABLE
        print(f"✅ negf_core CPP_NEGF_AVAILABLE = {CPP_NEGF_AVAILABLE}")
        
    except ImportError as e:
        print(f"❌ negf_core import failed: {e}")
        return False
    
    try:
        # Test a simple C++ function call
        H = np.array([[0.1+0j, 0.05],[0.05, -0.1]], dtype=np.complex128)
        H00 = np.array([[0.0+0j, 0.05],[0.05, 0.0]], dtype=np.complex128)
        H01 = np.array([[0.0+0j, 0.02],[0.02, 0.0]], dtype=np.complex128)
        
        gs = cpp_negf.surface_gf(H00, H01, 0.0, 1e-6)
        print(f"✅ C++ surface_gf call successful, result shape: {np.array(gs).shape}")
        
        T = cpp_negf.transmission_from_leads(H, H00, H01, np.eye(2, dtype=np.complex128), 
                                           H00, H01, np.eye(2, dtype=np.complex128), 0.0, 1e-6)
        print(f"✅ C++ transmission_from_leads call successful, T = {float(T):.2e}")
        
    except Exception as e:
        print(f"❌ C++ function call failed: {e}")
        return False
        
    print("✅ All C++ backend tests passed!")
    return True

if __name__ == "__main__":
    success = test_cpp_backend()
    sys.exit(0 if success else 1)