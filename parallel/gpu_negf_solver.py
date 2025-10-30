"""
GPU-Native SCF NEGF Solver with comprehensive array type handling.
This module provides a complete GPU-accelerated SCF implementation that handles
all NumPy/CuPy array conversions automatically and robustly.
"""

import numpy as np
import warnings
from typing import Dict, Any, Tuple, Optional, List, Union

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def to_gpu_array(arr, dtype=None):
    """Convert array to GPU (CuPy) with proper error handling."""
    if not GPU_AVAILABLE:
        return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
    
    if arr is None:
        return None
    
    try:
        # Handle scalar values
        if np.isscalar(arr):
            return cp.asarray(arr, dtype=dtype) if dtype else cp.asarray(arr)
        
        # Handle arrays
        if hasattr(arr, 'get'):  # Already CuPy array
            return arr.astype(dtype) if dtype else arr
        else:  # NumPy array or list
            return cp.asarray(arr, dtype=dtype) if dtype else cp.asarray(arr)
    except Exception as e:
        warnings.warn(f"GPU conversion failed, using CPU: {e}")
        return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)

def to_cpu_array(arr, dtype=None):
    """Convert array to CPU (NumPy) with proper error handling."""
    if arr is None:
        return None
    
    try:
        # Handle scalar values
        if np.isscalar(arr):
            return float(arr) if dtype is None else dtype(arr)
        
        # Handle arrays
        if hasattr(arr, 'get'):  # CuPy array
            result = arr.get()
            return result.astype(dtype) if dtype else result
        else:  # Already NumPy array
            return arr.astype(dtype) if dtype else np.asarray(arr)
    except Exception as e:
        warnings.warn(f"CPU conversion failed: {e}")
        return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)

class GPUNEGFSolver:
    """GPU-native NEGF solver with automatic array type management."""
    
    def __init__(self, H_device, H_leads, V_couplings, eta=1e-6, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.eta = eta
        
        if self.use_gpu:
            # Clear GPU memory and prepare for new computation
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            
            # Convert all matrices to GPU
            self.H_device = to_gpu_array(H_device, dtype=complex)
            self.V_couplings = [to_gpu_array(V, dtype=complex) for V in V_couplings]
            
            # Convert lead matrices
            self.H_leads = []
            for lead in H_leads:
                lead_gpu = {
                    'H00': to_gpu_array(lead['H00'], dtype=complex),
                    'H01': to_gpu_array(lead['H01'], dtype=complex)
                }
                self.H_leads.append(lead_gpu)
        else:
            # CPU mode
            self.H_device = np.asarray(H_device, dtype=complex)
            self.V_couplings = [np.asarray(V, dtype=complex) for V in V_couplings]
            self.H_leads = []
            for lead in H_leads:
                lead_cpu = {
                    'H00': np.asarray(lead['H00'], dtype=complex),
                    'H01': np.asarray(lead['H01'], dtype=complex)
                }
                self.H_leads.append(lead_cpu)
        
        self.n_device = self.H_device.shape[0]
        self.n_leads = len(H_leads)
    
    def _get_backend_modules(self):
        """Get appropriate backend modules (NumPy or CuPy)."""
        if self.use_gpu and GPU_AVAILABLE:
            return cp, cp.linalg
        else:
            return np, np.linalg
    
    def surface_greens_function(self, energy: float, lead_idx: int):
        """Compute surface Green's function using iterative method."""
        xp, linalg = self._get_backend_modules()
        
        lead = self.H_leads[lead_idx]
        H00 = lead['H00']
        H01 = lead['H01']
        
        # Handle dimension mismatch
        if H01.shape[1] != H00.shape[0]:
            if self.use_gpu:
                H01_padded = cp.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
                H01_padded[:H01.shape[0], :H01.shape[1]] = H01
                H01 = H01_padded
            else:
                H01_padded = np.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
                H01_padded[:H01.shape[0], :H01.shape[1]] = H01
                H01 = H01_padded
        
        # Energy matrix
        E = (energy + 1j * self.eta) * xp.eye(H00.shape[0], dtype=complex)
        
        # Hermitian conjugate
        H10 = H01.T.conj()
        
        # Iterative calculation
        alpha = H01.copy()
        beta = H10.copy()
        eps_s = H00.copy()
        
        max_iter = 100
        tol = 1e-12
        
        for i in range(max_iter):
            try:
                g_bulk = linalg.inv(E - eps_s)
            except Exception:
                g_bulk = linalg.pinv(E - eps_s)
            
            # Update
            alpha_new = alpha @ g_bulk @ alpha
            beta_new = beta @ g_bulk @ beta
            
            # Check convergence
            alpha_diff = xp.max(xp.abs(alpha_new - alpha))
            beta_diff = xp.max(xp.abs(beta_new - beta))
            
            if max(float(alpha_diff), float(beta_diff)) < tol:
                break
            
            # Update for next iteration
            eps_s = eps_s + alpha @ g_bulk @ beta + beta @ g_bulk @ alpha
            alpha = alpha_new
            beta = beta_new
        
        # Final surface Green's function
        try:
            g_s = linalg.inv(E - eps_s)
        except Exception:
            g_s = linalg.pinv(E - eps_s)
        
        return g_s
    
    def self_energy(self, energy: float, lead_idx: int):
        """Compute self-energy for a lead."""
        g_s = self.surface_greens_function(energy, lead_idx)
        V_coupling = self.V_couplings[lead_idx]
        
        # Self-energy: Σ = V† G_s V
        Sigma = V_coupling.T.conj() @ g_s @ V_coupling
        
        return Sigma
    
    def retarded_greens_function(self, energy: float):
        """Compute retarded Green's function."""
        xp, linalg = self._get_backend_modules()
        
        # Energy matrix
        E = (energy + 1j * self.eta) * xp.eye(self.n_device, dtype=complex)
        
        # Effective Hamiltonian
        H_eff = self.H_device.copy()
        
        # Add self-energies
        for i in range(self.n_leads):
            Sigma = self.self_energy(energy, i)
            
            # Ensure dimensions match
            if Sigma.shape != (self.n_device, self.n_device):
                if self.use_gpu:
                    Sigma_full = cp.zeros((self.n_device, self.n_device), dtype=complex)
                else:
                    Sigma_full = np.zeros((self.n_device, self.n_device), dtype=complex)
                
                min_dim = min(Sigma.shape[0], self.n_device)
                if i == 0:  # Left lead
                    Sigma_full[:min_dim, :min_dim] = Sigma[:min_dim, :min_dim]
                else:  # Right lead
                    Sigma_full[-min_dim:, -min_dim:] = Sigma[:min_dim, :min_dim]
                Sigma = Sigma_full
            
            H_eff += Sigma
        
        # Green's function
        try:
            G_r = linalg.inv(E - H_eff)
        except Exception:
            G_r = linalg.pinv(E - H_eff)
        
        return G_r
    
    def advanced_greens_function(self, energy: float):
        """Compute advanced Green's function."""
        G_r = self.retarded_greens_function(energy)
        return G_r.T.conj()
    
    def gamma_matrix(self, energy: float, lead_idx: int):
        """Compute gamma matrix."""
        Sigma_r = self.self_energy(energy, lead_idx)
        Sigma_a = Sigma_r.T.conj()
        
        Gamma = 1j * (Sigma_r - Sigma_a)
        return Gamma
    
    def transmission(self, energy: float, lead_i: int = 0, lead_j: int = 1) -> float:
        """Compute transmission coefficient."""
        xp, _ = self._get_backend_modules()
        
        G_r = self.retarded_greens_function(energy)
        G_a = self.advanced_greens_function(energy)
        
        Gamma_i = self.gamma_matrix(energy, lead_i)
        Gamma_j = self.gamma_matrix(energy, lead_j)
        
        # Transmission
        T_matrix = Gamma_i @ G_r @ Gamma_j @ G_a
        T = xp.real(xp.trace(T_matrix))
        
        # Convert to Python float
        return float(T)
    
    def verify_gpu_usage(self):
        """Verify that GPU arrays are being used."""
        if self.use_gpu and GPU_AVAILABLE:
            gpu_memory = cp.get_default_memory_pool().used_bytes() / 1024**2  # MB
            if gpu_memory > 10:  # More than 10 MB indicates active GPU usage
                return True, f"GPU active: {gpu_memory:.1f} MB used"
            else:
                return False, f"GPU minimal usage: {gpu_memory:.1f} MB"
        return False, "GPU not available"
    
    def lesser_greens_function(self, energy: float, mu_leads: List[float], temperature: float = 0.0):
        """Compute lesser Green's function."""
        xp, _ = self._get_backend_modules()
        
        G_r = self.retarded_greens_function(energy)
        G_a = self.advanced_greens_function(energy)
        
        # Fermi function
        def fermi(E, mu, T):
            if T <= 0:
                return 1.0 if E <= mu else 0.0
            x = (E - mu) / (8.617333262e-5 * T)
            if x > 50:
                return 0.0
            elif x < -50:
                return 1.0
            else:
                if x >= 0:
                    return float(np.exp(-x) / (1.0 + np.exp(-x)))
                else:
                    return float(1.0 / (1.0 + np.exp(x)))
        
        # Initialize Sigma_lesser
        if self.use_gpu:
            Sigma_lesser = cp.zeros_like(self.H_device, dtype=complex)
        else:
            Sigma_lesser = np.zeros_like(self.H_device, dtype=complex)
        
        # Sum contributions from leads
        for i, mu in enumerate(mu_leads):
            if i >= self.n_leads:
                break
            
            f = fermi(energy, mu, temperature)
            Gamma = self.gamma_matrix(energy, i)
            
            Sigma_lesser += 1j * f * Gamma
        
        # Lesser Green's function
        G_lesser = G_r @ Sigma_lesser @ G_a
        return G_lesser
    
    def local_charge_density(self, energy_grid, mu_leads: List[float], temperature: float = 0.0):
        """Compute local charge density."""
        xp, _ = self._get_backend_modules()
        
        if self.use_gpu:
            rho = cp.zeros(self.n_device, dtype=float)
        else:
            rho = np.zeros(self.n_device, dtype=float)
        
        for energy in energy_grid:
            G_lesser = self.lesser_greens_function(float(energy), mu_leads, temperature)
            
            # Extract diagonal and take imaginary part
            diag_elements = xp.imag(xp.diag(G_lesser))
            rho += diag_elements
        
        # Integration
        dE = float(energy_grid[1] - energy_grid[0]) if len(energy_grid) > 1 else 1.0
        rho *= -dE / np.pi
        
        # Always return CPU array for SCF compatibility
        return to_cpu_array(rho, dtype=float)
    
    def charge_density_at_bias(self, mu_left: float, mu_right: float, 
                              temperature: float = 0.0, E_span: float = 0.5, n_points: int = 101):
        """Compute charge density under bias."""
        # Create energy grid (always NumPy for compatibility)
        mu_min = min(mu_left, mu_right) - E_span/2
        mu_max = max(mu_left, mu_right) + E_span/2
        energy_grid = np.linspace(mu_min, mu_max, n_points)
        
        # Chemical potentials
        mu_leads = [mu_left, mu_right] if self.n_leads >= 2 else [mu_left]
        
        return self.local_charge_density(energy_grid, mu_leads, temperature)