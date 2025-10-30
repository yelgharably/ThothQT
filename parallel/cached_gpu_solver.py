"""
Optimized GPU-enabled SCF NEGF solver with instance caching and performance optimization.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from gpu_negf_solver import GPUNEGFSolver
import hashlib

class CachedGPUNEGFSolver:
    """
    Cached GPU NEGF solver that reuses instances for performance.
    """
    
    _cache = {}  # Class-level cache
    
    def __init__(self, H_device, H_leads, V_couplings, use_gpu=True, eta=1e-6):
        self.use_gpu = use_gpu
        self.eta = eta
        
        # Create a hash of the matrices to use as cache key
        cache_key = self._create_cache_key(H_device, H_leads, V_couplings, eta, use_gpu)
        
        # Check if we already have this solver cached
        if cache_key in self._cache:
            self.solver = self._cache[cache_key]
        else:
            # Create new solver and cache it
            self.solver = GPUNEGFSolver(H_device, H_leads, V_couplings, use_gpu=use_gpu, eta=eta)
            self._cache[cache_key] = self.solver
            
            # Limit cache size to prevent memory issues
            if len(self._cache) > 5:  # Keep only 5 most recent solvers
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
    
    def _create_cache_key(self, H_device, H_leads, V_couplings, eta, use_gpu):
        """Create a hash key for caching based on matrix shapes and parameters."""
        # Use matrix shapes and key parameters for cache key
        key_data = f"{H_device.shape}_{len(H_leads)}_{len(V_couplings)}_{eta}_{use_gpu}"
        
        # Add a simple checksum of matrix data (first few elements)
        if hasattr(H_device, 'flatten'):
            flat_device = H_device.flatten()
            if len(flat_device) > 0:
                key_data += f"_{complex(flat_device[0])}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def charge_density_at_bias(self, mu_left, mu_right, temperature=0.0, E_span=0.5, n_points=101):
        """Compute charge density using cached GPU solver."""
        return self.solver.charge_density_at_bias(mu_left, mu_right, temperature, E_span, n_points)
    
    def transmission(self, energy, lead_i=0, lead_j=1):
        """Compute transmission using cached GPU solver."""
        return self.solver.transmission(energy, lead_i, lead_j)
    
    @classmethod
    def clear_cache(cls):
        """Clear the solver cache."""
        cls._cache.clear()


def get_cached_gpu_negf_solver(H_device, H_leads, V_couplings, use_gpu=True, eta=1e-6):
    """
    Factory function to get a cached GPU NEGF solver.
    This avoids recreating solvers unnecessarily.
    """
    return CachedGPUNEGFSolver(H_device, H_leads, V_couplings, use_gpu=use_gpu, eta=eta)