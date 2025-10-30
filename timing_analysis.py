#!/usr/bin/env python3
"""
Debug script to trace exactly where time is being spent in the main workflow
"""

import time
import contextlib
import sys
import os

# Add cpp directory  
sys.path.insert(0, 'cpp')

@contextlib.contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    print(f'{label}: {(end-start)*1000:.1f} ms')

def trace_main_workflow():
    print('=== Main Workflow Timing Analysis ===')
    
    with timer('Importing modules'):
        from negf_core import NEGFSolver, extract_kwant_matrices, CPP_NEGF_AVAILABLE
        from scf_solver import SCFSolver
        import kwant
        import numpy as np
        
    print(f'CPP_NEGF_AVAILABLE: {CPP_NEGF_AVAILABLE}')
    
    # Create a simple graphene-like system (similar to main script but smaller)
    with timer('Building Kwant system'):
        # Simple tight-binding chain for testing
        lat = kwant.lattice.chain(a=1.0, norbs=2)
        
        def onsite(site, t):
            return np.eye(2) * 0.1
            
        def hopping(site1, site2, t):
            return -np.eye(2) * t
            
        # Build system
        sys_builder = kwant.Builder()
        
        # Add sites (small system)
        for i in range(20):  # 20 sites = 40 orbitals
            sys_builder[lat(i)] = onsite
            
        # Add hoppings
        for i in range(19):
            sys_builder[lat(i), lat(i+1)] = hopping
            
        # Add leads
        lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
        lead[lat(0)] = onsite
        lead[lat(0), lat(1)] = hopping
        
        sys_builder.attach_lead(lead)
        sys_builder.attach_lead(lead.reversed())
        
        fsys = sys_builder.finalized()
    
    params = {'t': 2.7}
    energy = 0.0
    
    with timer('Extracting Kwant matrices'):
        H_device, H_leads, V_couplings = extract_kwant_matrices(fsys, energy, params=params)
        
    print(f'System sizes: H_device {H_device.shape}, H_leads: {[lead["H00"].shape for lead in H_leads]}, V_couplings: {[V.shape for V in V_couplings]}')
    
    with timer('Creating NEGFSolver'):
        negf = NEGFSolver(H_device, H_leads, V_couplings)
        
    with timer('Single transmission calculation'):
        T = negf.transmission(energy)
        
    print(f'Transmission: {T:.2e}')
    
    # Test SCF workflow components
    with timer('Creating SCFSolver'):
        lattice_sites = np.array([[i, 0] for i in range(20)])  # 20 sites
        scf = SCFSolver(lattice_sites)
        
    with timer('Single charge density calculation'):
        try:
            charge_density = scf.compute_charge_density(fsys, params, 0.0005, -0.0005, temperature=0.0)
            print(f'Charge density computed: {len(charge_density)} sites')
        except Exception as e:
            print(f'Charge density failed: {e}')
            
    with timer('Single Poisson solve'):
        try:
            # Dummy charge density
            rho = np.random.random(20) * 1e-3
            phi = scf.poisson_solver.solve_graphene(rho, 0.001, (0.0005, -0.0005))
            print(f'Poisson solve: {len(phi) if phi is not None else None} sites')
        except Exception as e:
            print(f'Poisson solve failed: {e}')

if __name__ == '__main__':
    trace_main_workflow()