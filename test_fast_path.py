#!/usr/bin/env python3
"""
Quick test to see if the C++ backend fast path is actually being used
"""

import os
import sys
sys.path.insert(0, 'cpp')

# Enable debug mode by setting environment variable
os.environ['NEGFSW_DEBUG'] = '1'

from negf_core import NEGFSolver, CPP_NEGF_AVAILABLE
import numpy as np

print(f'CPP_NEGF_AVAILABLE: {CPP_NEGF_AVAILABLE}')

# Test with simple matrices to see if C++ path is taken
H_device = np.array([[0.1, 0.05], [0.05, -0.1]], dtype=complex)
H_leads = [
    {'H00': np.array([[0.0, 0.05], [0.05, 0.0]], dtype=complex),
     'H01': np.array([[0.0, 0.02], [0.02, 0.0]], dtype=complex)},
    {'H00': np.array([[0.0, 0.05], [0.05, 0.0]], dtype=complex),  
     'H01': np.array([[0.0, 0.02], [0.02, 0.0]], dtype=complex)}
]
V_couplings = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
]

negf = NEGFSolver(H_device, H_leads, V_couplings)

print('\nTesting transmission calculation...')
T = negf.transmission(0.0, 0, 1)
print(f'Transmission: {T}')

print('\nTesting finite bias current...')
I = negf.finite_bias_current_cpp(0.005, -0.005, 300.0, 0.0)
print(f'Finite bias current: {I}')