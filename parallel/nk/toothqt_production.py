"""
ThoothQT (Quantum Transport Library)
=====================================

A production-ready NEGF (Non-Equilibrium Green's Function) implementation for 
quantum transport calculations with GPU acceleration support.

Features:
- Sancho-Rubio decimation with numerical stabilization
- Analytical 1D solution for perfect accuracy
- Fisher-Lee transmission formula
- Landauer current calculation
- GPU/CPU backend with automatic fallback
- Sparse matrix support for large systems

Author: Quantum Sensing Project
Version: 2.0.0 (Production)
"""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Callable

# GPU backend support (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    import cupyx.scipy.sparse.linalg as cpx_spla
    _CUPY_AVAILABLE = (cp.cuda.runtime.getDeviceCount() > 0)
except Exception:
    cp, cpx_sp, cpx_spla = None, None, None
    _CUPY_AVAILABLE = False

# Physical constants
KB_EV = 8.617333262145e-5  # Boltzmann constant in eV/K
E_CHARGE = 1.602176634e-19  # Elementary charge in C
H_PLANCK = 6.62607015e-34   # Planck constant in J·s
G0_SI = 2 * E_CHARGE**2 / H_PLANCK  # Conductance quantum in S


def fermi(E: Union[float, np.ndarray], mu: float, kT: float, 
          gpu: bool = False) -> Union[float, np.ndarray]:
    """
    Fermi-Dirac distribution function with overflow protection.
    
    f(E) = 1 / (1 + exp((E - mu) / kT))
    
    Parameters
    ----------
    E : float or array
        Energy values (eV)
    mu : float
        Chemical potential (eV)
    kT : float
        Thermal energy (eV)
    gpu : bool
        Use GPU if available
        
    Returns
    -------
    f : float or array
        Fermi-Dirac occupation at energy E
    """
    x = (E - mu) / max(kT, 1e-12)
    
    if gpu and _CUPY_AVAILABLE:
        xp = cp
    else:
        xp = np
        x = np.asarray(x)
    
    # Overflow protection
    out = xp.empty_like(x, dtype=float)
    pos = x > 50   # exp(50) ~ 5e21, negligible occupation
    neg = x < -50  # exp(-50) ~ 2e-22, full occupation
    mid = ~(pos | neg)
    
    out[pos] = 0.0
    out[neg] = 1.0
    out[mid] = 1.0 / (1.0 + xp.exp(x[mid]))
    
    return out


@dataclass
class PeriodicLead:
    """
    Semi-infinite periodic lead definition.
    
    Attributes
    ----------
    H00 : array or sparse matrix
        On-site Hamiltonian of lead unit cell (m × m)
    H01 : array or sparse matrix
        Hopping from unit cell n to n+1 (m × m)
    tau_cpl : array or sparse matrix, optional
        Explicit coupling matrix to device (m × m)
        If None, uses H01 as coupling
    """
    H00: Union[np.ndarray, sp.spmatrix]
    H01: Union[np.ndarray, sp.spmatrix]
    tau_cpl: Optional[Union[np.ndarray, sp.spmatrix]] = None
    
    def __post_init__(self):
        """Validate lead dimensions."""
        if self.H00.shape[0] != self.H00.shape[1]:
            raise ValueError(f"H00 must be square, got shape {self.H00.shape}")
        if self.H01.shape != self.H00.shape:
            raise ValueError(f"H01 shape {self.H01.shape} must match H00 shape {self.H00.shape}")
        if self.tau_cpl is not None and self.tau_cpl.shape != self.H00.shape:
            raise ValueError(f"tau_cpl shape {self.tau_cpl.shape} must match H00 shape {self.H00.shape}")


class SanchoRubioDecimator:
    """
    Sancho-Rubio iterative decimation for surface Green's function.
    
    Computes the surface Green's function of a semi-infinite periodic lead:
        g_s = lim_{n→∞} [E·I - H_00 - H_01·g_s·H_01†]^{-1}
    
    References
    ----------
    M. P. López Sancho et al., J. Phys. F: Met. Phys. 14, 1205 (1984)
    M. P. López Sancho et al., J. Phys. F: Met. Phys. 15, 851 (1985)
    """
    
    def __init__(self, lead: PeriodicLead, eta: float = 1e-6,
                 max_iter: int = 1000, tol: float = 1e-12, gpu: bool = False):
        """
        Initialize Sancho-Rubio decimator.
        
        Parameters
        ----------
        lead : PeriodicLead
            Lead definition
        eta : float
            Small imaginary part for regularization (eV)
        max_iter : int
            Maximum decimation iterations
        tol : float
            Convergence tolerance (Frobenius norm)
        gpu : bool
            Use GPU if available
        """
        self.lead = lead
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.gpu = gpu and _CUPY_AVAILABLE
        
        if gpu and not _CUPY_AVAILABLE:
            warnings.warn("GPU requested but CuPy unavailable. Using CPU.")
            self.gpu = False
        
        # Set backend
        self.xp = cp if self.gpu else np
        self.sp = cpx_sp if self.gpu else sp
        
        # Convert lead matrices to backend
        self.H00 = self._to_array(lead.H00)
        self.H01 = self._to_array(lead.H01)
        self.tau_cpl = self._to_array(lead.tau_cpl) if lead.tau_cpl is not None else None
        
        # Detect 1D system (single site per unit cell)
        self.is_1d = (self.H00.shape[0] == 1)
    
    def _detect_1d_system(self) -> bool:
        """Detect if system is 1D (single orbital per unit cell)."""
        return self.H00.shape[0] == 1
    
    def _to_array(self, arr: Optional[Union[np.ndarray, sp.spmatrix]]) -> Optional[np.ndarray]:
        """
        Convert input to dense array on current backend.
        
        Parameters
        ----------
        arr : array or sparse matrix or None
            Input array
            
        Returns
        -------
        arr_backend : ndarray or None
            Dense array on current backend (CPU/GPU)
        """
        if arr is None:
            return None
        
        # Convert sparse to dense
        if sp.issparse(arr):
            arr = arr.toarray()
        
        # Convert to backend (numpy/cupy)
        return self.xp.asarray(arr, dtype=complex)
    
    def _inv(self, arr: np.ndarray) -> np.ndarray:
        """Matrix inversion on current backend."""
        return self.xp.linalg.inv(arr)
    
    def surface_g_1d_stable(self, E: float) -> np.ndarray:
        """
        Compute surface Green's function for 1D using direct formula.
        
        For 1D tight-binding: g_s = [E - ε0 - t²·g_s]^{-1}
        Solution: g_s = (E - ε0 ± sqrt((E-ε0)² - 4t²)) / (2t²)
        
        Choose sign to ensure Im[g_s] < 0 (causality).
        
        Parameters
        ----------
        E : float
            Energy (eV)
            
        Returns
        -------
        g_s : complex
            Surface Green's function (scalar for 1D)
        """
        xp = self.xp
        
        # Extract parameters
        eps0 = self.H00[0, 0]  # On-site energy
        t = self.H01[0, 0]      # Hopping
        
        # Complex energy
        z = E + 1j * self.eta
        
        # Discriminant
        delta = (z - eps0)**2 - 4 * t * xp.conj(t)
        sqrt_delta = xp.sqrt(delta)
        
        # Two solutions
        g1 = ((z - eps0) + sqrt_delta) / (2 * t * xp.conj(t))
        g2 = ((z - eps0) - sqrt_delta) / (2 * t * xp.conj(t))
        
        # Choose solution with Im[g] < 0 (retarded)
        if xp.imag(g1) < xp.imag(g2):
            g_s = g1
        else:
            g_s = g2
        
        # Return as 1×1 matrix
        return xp.array([[g_s]], dtype=complex)
    
    def surface_g(self, E: float) -> np.ndarray:
        """
        Compute surface Green's function using stabilized Sancho-Rubio decimation.
        
        Parameters
        ----------
        E : float
            Energy (eV)
            
        Returns
        -------
        g_s : ndarray (m × m, complex)
            Surface Green's function of semi-infinite lead
            
        Algorithm
        ---------
        Stabilized Sancho-Rubio with:
        1. Adaptive tolerance based on matrix norms
        2. Overflow detection and recovery
        3. Rescaling when matrix norms grow too large
        4. Increased eta near band edges
        5. 1D systems use analytical formula for stability
        """
        # Use analytical formula for 1D systems (much more stable)
        if self.is_1d:
            return self.surface_g_1d_stable(E)
        
        xp = self.xp
        H00, H01 = self.H00, self.H01
        
        # Adaptive eta: increase near band edges for stability
        # Estimate band edge from Frobenius norm
        h_norm = xp.linalg.norm(H00 + H01 + H01.conj().T, ord='fro')
        band_edge_estimate = h_norm / xp.sqrt(H00.shape[0])
        
        # If close to band edge, use larger eta
        if abs(abs(E) - band_edge_estimate) < 0.1:
            eta_adaptive = max(self.eta * 10, 1e-2)
        else:
            eta_adaptive = self.eta
        
        # Energy with adaptive regularization
        zI = (E + 1j * eta_adaptive) * xp.eye(H00.shape[0], dtype=complex)
        
        # Initial values
        a = H00.copy()
        alpha = H01.copy()
        beta = H01.conj().T.copy()
        
        # Rescaling factor to prevent overflow
        scale = 1.0
        max_norm = 1e10  # Threshold for rescaling
        
        try:
            g = self._inv(zI - a)
        except:
            # If initial inversion fails, use larger eta
            zI = (E + 1j * max(eta_adaptive * 100, 0.1)) * xp.eye(H00.shape[0], dtype=complex)
            g = self._inv(zI - a)
        
        # Decimation iteration with stabilization
        for iteration in range(self.max_iter):
            # Check for overflow before computation
            alpha_norm = xp.linalg.norm(alpha, ord='fro')
            beta_norm = xp.linalg.norm(beta, ord='fro')
            g_norm = xp.linalg.norm(g, ord='fro')
            
            # Rescale if matrices growing too large
            if alpha_norm > max_norm or beta_norm > max_norm or g_norm > max_norm:
                rescale = 1.0 / xp.sqrt(alpha_norm * beta_norm)
                alpha = alpha * rescale
                beta = beta * rescale
                scale = scale / rescale
                
            # Check for NaN/Inf
            if xp.any(xp.isnan(g)) or xp.any(xp.isinf(g)):
                warnings.warn(f"Sancho-Rubio: NaN/Inf detected at E={E:.6f} eV, iteration {iteration}")
                # Return best guess with large eta
                zI_safe = (E + 1j * 0.5) * xp.eye(H00.shape[0], dtype=complex)
                return self._inv(zI_safe - H00)
            
            # Update matrices with stabilized computation
            try:
                # Compute updates
                agb = alpha @ g @ beta
                aga = alpha @ g @ alpha
                bgb = beta @ g @ beta
                
                a_new = a + agb
                alpha_new = aga
                beta_new = bgb
                
                # Invert to get new Green's function
                g_new = self._inv(zI - a_new)
                
            except Exception as e:
                warnings.warn(f"Sancho-Rubio: Matrix operation failed at E={E:.6f} eV: {e}")
                return g  # Return current best estimate
            
            # Adaptive convergence check
            # Use relative difference to handle different energy scales
            diff_norm = xp.linalg.norm(g_new - g, ord='fro')
            g_norm = xp.linalg.norm(g, ord='fro')
            rel_diff = diff_norm / (g_norm + 1e-10)
            
            # Check alpha/beta convergence (often better indicator)
            alpha_diff = xp.linalg.norm(alpha_new - alpha, ord='fro')
            alpha_conv = alpha_diff / (xp.linalg.norm(alpha, ord='fro') + 1e-10)
            
            beta_diff = xp.linalg.norm(beta_new - beta, ord='fro')
            beta_conv = beta_diff / (xp.linalg.norm(beta, ord='fro') + 1e-10)
            
            # Converged if both Green's function and coupling matrices stable
            if rel_diff < self.tol and alpha_conv < self.tol and beta_conv < self.tol:
                return g_new * scale
            
            # Alternative: alpha/beta very small means lead decoupled
            if xp.linalg.norm(alpha_new, ord='fro') < self.tol * 1e-3 and \
               xp.linalg.norm(beta_new, ord='fro') < self.tol * 1e-3:
                return g_new * scale
            
            # Prepare next iteration
            a, alpha, beta, g = a_new, alpha_new, beta_new, g_new
        
        # Did not converge - use safer fallback
        warnings.warn(f"Sancho-Rubio did not converge after {self.max_iter} iterations "
                     f"at E={E:.6f} eV (rel_diff: {rel_diff:.2e})")
        
        # Fallback: return current estimate or compute with very large eta
        if xp.any(xp.isnan(g)) or xp.any(xp.isinf(g)):
            zI_safe = (E + 1j * 1.0) * xp.eye(H00.shape[0], dtype=complex)
            return self._inv(zI_safe - H00)
        
        return g * scale
    
    def sigma(self, E: float) -> np.ndarray:
        """
        Compute lead self-energy.
        
        Sigma(E) = tau† · g_s(E) · tau
        
        where g_s is the surface Green's function and tau is the
        coupling matrix between lead and device.
        
        Parameters
        ----------
        E : float
            Energy (eV)
            
        Returns
        -------
        Sigma : ndarray (m × m, complex)
            Self-energy matrix
        """
        g_s = self.surface_g(E)
        
        # Use explicit coupling if provided, otherwise H01
        if self.tau_cpl is not None:
            tau = self.tau_cpl
        else:
            tau = self.H01
        
        # Sigma = tau† · g_s · tau
        Sigma = tau.conj().T @ g_s @ tau
        return Sigma


@dataclass
class Device:
    """
    Central device region connected to leads.
    
    Attributes
    ----------
    H : sparse matrix (N × N)
        Device Hamiltonian
    S : sparse matrix (N × N) or None
        Overlap matrix (None for orthogonal basis)
    left : PeriodicLead
        Left lead definition
    right : PeriodicLead
        Right lead definition
    Ef : float
        Fermi energy (eV)
    """
    H: sp.csr_matrix
    S: Optional[sp.csr_matrix]
    left: PeriodicLead
    right: PeriodicLead
    Ef: float = 0.0
    
    def __post_init__(self):
        """Validate device dimensions."""
        N = self.H.shape[0]
        if self.H.shape[1] != N:
            raise ValueError(f"Device Hamiltonian must be square, got {self.H.shape}")
        if self.S is not None and self.S.shape != (N, N):
            raise ValueError(f"Overlap matrix shape {self.S.shape} must match H shape {self.H.shape}")
        
        m_left = self.left.H00.shape[0]
        m_right = self.right.H00.shape[0]
        if N < m_left + m_right:
            raise ValueError(f"Device size {N} too small for leads (left: {m_left}, right: {m_right})")


class LinearSolver:
    """Abstract base class for linear system solvers."""
    def solve(self, A, b):
        raise NotImplementedError("LinearSolver is an abstract base class.")


class CPUSolver(LinearSolver):
    """
    CPU sparse direct solver using SuperLU.
    
    Fast for small to medium systems (N < 10,000).
    """
    
    def __init__(self, A_csr: sp.csr_matrix):
        """
        Initialize solver with LU factorization.
        
        Parameters
        ----------
        A_csr : sparse matrix
            System matrix (will be converted to CSC for SuperLU)
        """
        self.lu = spla.splu(A_csr.tocsc())
    
    def solve(self, A_unused, b: np.ndarray) -> np.ndarray:
        """
        Solve A·x = b using precomputed LU factorization.
        
        Parameters
        ----------
        A_unused : ignored
            Placeholder for API compatibility
        b : ndarray (N, m)
            Right-hand side
            
        Returns
        -------
        x : ndarray (N, m)
            Solution
        """
        return self.lu.solve(b)


class GPUSolver(LinearSolver):
    """
    GPU iterative solver using GMRES with diagonal preconditioner.
    
    Efficient for large sparse systems (N > 10,000) with GPUs.
    """
    
    def __init__(self, A_csr, tol: float = 1e-9, maxiter: Optional[int] = None):
        """
        Initialize GPU solver.
        
        Parameters
        ----------
        A_csr : sparse matrix (GPU)
            System matrix
        tol : float
            GMRES convergence tolerance
        maxiter : int or None
            Maximum GMRES iterations (None = auto)
        """
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available for GPU computations.")
        
        self.A = A_csr
        self.tol = tol
        self.maxiter = maxiter if maxiter is not None else A_csr.shape[0]
        
        # Diagonal preconditioner: M = diag(A)
        D = self.A.diagonal()
        D = cp.where(cp.abs(D) > 1e-30, D, 1.0 + 0j)
        self.Dinv = 1.0 / D
    
    def _apply_preconditioner(self, r):
        """Apply diagonal preconditioner: M^{-1} · r"""
        return self.Dinv * r
    
    def solve(self, A_unused, b: cp.ndarray) -> cp.ndarray:
        """
        Solve A·x = b using GMRES.
        
        Parameters
        ----------
        A_unused : ignored
            Placeholder for API compatibility
        b : ndarray (N, m) on GPU
            Right-hand side
            
        Returns
        -------
        x : ndarray (N, m) on GPU
            Solution
        """
        n, m = b.shape
        x = cp.zeros((n, m), dtype=b.dtype)
        
        # Linear operator for A
        Aop = cpx_spla.LinearOperator(self.A.shape, matvec=lambda v: self.A.dot(v))
        
        # Preconditioner operator
        Mop = cpx_spla.LinearOperator(self.A.shape, matvec=self._apply_preconditioner)
        
        # Solve for each column
        for j in range(m):
            bj = b[:, j]
            xj, info = cpx_spla.gmres(Aop, bj, M=Mop, tol=self.tol, maxiter=self.maxiter)
            
            if info != 0:
                warnings.warn(f"GMRES did not converge for column {j}, info={info}")
            
            x[:, j] = xj  # Store solution directly (not preconditioned)
        
        return x


class EnergyMesh:
    """
    Energy grid with optional refinement around specific points.
    
    Useful for adaptive integration when transmission varies rapidly.
    """
    
    def __init__(self, Emin: float, Emax: float, n: int,
                 refine_at: Tuple[float, ...] = (), refine_pts: int = 200):
        """
        Create energy grid.
        
        Parameters
        ----------
        Emin, Emax : float
            Energy range (eV)
        n : int
            Number of base grid points
        refine_at : tuple of floats
            Energy points to refine around (e.g., Fermi level)
        refine_pts : int
            Number of refinement points per refine_at location
        """
        base = np.linspace(Emin, Emax, n)
        extra = []
        
        for E0 in refine_at:
            # Add dense grid around E0
            extra.append(np.linspace(max(Emin, E0 - 0.1), 
                                     min(Emax, E0 + 0.1), 
                                     refine_pts))
        
        # Combine and remove duplicates
        if extra:
            self.grid = np.unique(np.concatenate([base] + extra))
        else:
            self.grid = base
    
    def __len__(self):
        return len(self.grid)
    
    def __getitem__(self, idx):
        return self.grid[idx]


class NEGFEngine:
    """
    Non-Equilibrium Green's Function engine for quantum transport.
    
    Computes transmission, conductance, and current using the NEGF formalism:
        G(E) = [E·S - H - Σ_L(E) - Σ_R(E)]^{-1}
        T(E) = Tr[Γ_L · G · Γ_R · G†]
        I(V) = (2e²/h) ∫ T(E) [f_L(E) - f_R(E)] dE
    
    References
    ----------
    S. Datta, "Electronic Transport in Mesoscopic Systems" (1995)
    """
    
    def __init__(self, device: Device, Temp: float, eta: float = 1e-6, gpu: bool = False):
        """
        Initialize NEGF engine.
        
        Parameters
        ----------
        device : Device
            Device definition with leads
        Temp : float
            Temperature (K)
        eta : float
            Small imaginary part for Green's functions (eV)
        gpu : bool
            Use GPU if available
        """
        self.device = device
        self.Temp = Temp
        self.kT = Temp * KB_EV  # Thermal energy in eV
        self.eta = eta
        self.gpu = gpu and _CUPY_AVAILABLE
        self.backend = "gpu" if self.gpu else "cpu"
        
        if gpu and not _CUPY_AVAILABLE:
            warnings.warn("GPU requested but CuPy unavailable. Using CPU.")
            self.gpu = False
        
        # Set backend modules
        self.xp = cp if self.gpu else np
        self.spmod = cpx_sp if self.gpu else sp
        
        # Initialize lead decimators
        self.decL = SanchoRubioDecimator(device.left, eta=eta, gpu=self.gpu)
        self.decR = SanchoRubioDecimator(device.right, eta=eta, gpu=self.gpu)
        
        print(f"NEGF Engine initialized: {self.backend.upper()} mode, T={Temp:.1f}K")
    
    def _assemble_system_matrix(self, E: float, SigmaL: np.ndarray, 
                                SigmaR: np.ndarray) -> sp.csr_matrix:
        """
        Assemble NEGF system matrix.
        
        A(E) = E·S - H - Σ_L(E) - Σ_R(E)
        
        Parameters
        ----------
        E : float
            Energy (eV)
        SigmaL, SigmaR : ndarray
            Self-energies from left and right leads
            
        Returns
        -------
        A : sparse matrix (N × N)
            System matrix
        """
        H = self.device.H
        N = H.shape[0]
        m = self.device.left.H00.shape[0]  # Lead block size
        
        if self.backend == "cpu":
            # CPU: sparse matrices
            if self.device.S is None:
                # Don't add extra eta - self-energies already provide regularization
                A = E * sp.eye(N, format="csr", dtype=complex) - H.astype(complex)
            else:
                A = E * self.device.S.astype(complex) - H.astype(complex)
            
            # Add self-energies to corner blocks (use lil for efficiency)
            A = A.tolil()
            A[:m, :m] -= sp.csr_matrix(SigmaL, dtype=complex)
            A[-m:, -m:] -= sp.csr_matrix(SigmaR, dtype=complex)
            
            return A.tocsr()
        
        # GPU: sparse matrices on device
        xp, spm = self.xp, self.spmod
        
        if self.device.S is None:
            A = E * spm.identity(N, dtype=complex, format="csr")
            A = A - spm.csr_matrix(H.astype(complex))
        else:
            A = E * spm.csr_matrix(self.device.S.astype(complex))
            A = A - spm.csr_matrix(H.astype(complex))
        
        # Add self-energies (convert to lil for block updates)
        A = A.tolil()
        SL_gpu = xp.asarray(SigmaL, dtype=complex)
        SR_gpu = xp.asarray(SigmaR, dtype=complex)
        A[:m, :m] -= spm.csr_matrix(SL_gpu)
        A[-m:, -m:] -= spm.csr_matrix(SR_gpu)
        
        return A.tocsr()
    
    @staticmethod
    def _broadening(Sigma: np.ndarray) -> np.ndarray:
        """
        Compute broadening matrix from self-energy.
        
        Γ = i(Σ - Σ†)
        
        Parameters
        ----------
        Sigma : ndarray (m × m, complex)
            Self-energy
            
        Returns
        -------
        Gamma : ndarray (m × m, complex)
            Broadening matrix
        """
        return 1j * (Sigma - Sigma.conj().T)
    
    def transmission(self, E: float) -> float:
        """
        Compute transmission coefficient at energy E.
        
        Uses Fisher-Lee formula:
            T(E) = Tr[Γ_L · G · Γ_R · G†]
        
        where G is the retarded Green's function and Γ are broadening matrices.
        
        Parameters
        ----------
        E : float
            Energy (eV)
            
        Returns
        -------
        T : float
            Transmission coefficient (dimensionless, 0 ≤ T ≤ N_channels)
        """
        # Get self-energies from leads
        SigmaL = self.decL.sigma(E)
        SigmaR = self.decR.sigma(E)
        
        # Assemble system matrix
        A = self._assemble_system_matrix(E, SigmaL, SigmaR)
        
        m = self.device.left.H00.shape[0]  # Lead block size
        N = A.shape[0]
        
        if self.backend == "cpu":
            # CPU: Direct solve with SuperLU
            lu = spla.splu(A.tocsc())
            
            # Build RHS: identity on right contact
            b = np.zeros((N, m), dtype=complex)
            b[N - m:N, :] = np.eye(m, dtype=complex)
            
            # Solve A·G = b to get Green's function columns
            G_cols = lu.solve(b)  # Shape: (N, m)
            
            # Extract blocks needed for transmission
            G_LR = G_cols[:m, :]  # Top m rows (left-right block)
            G_RL = G_LR.conj().T   # Advanced GF: G^†
            
            # Broadening matrices
            GammaL = self._broadening(SigmaL)
            GammaR = self._broadening(SigmaR)
            
            # Fisher-Lee formula
            T = np.real(np.trace(GammaL @ G_LR @ GammaR @ G_RL))
            
            return float(max(T, 0.0))  # Ensure non-negative
        
        # GPU: Iterative solve with GMRES
        xp = self.xp
        
        # Build RHS on GPU
        b = xp.zeros((N, m), dtype=complex)
        b[N - m:N, :] = xp.eye(m, dtype=complex)
        
        # Solve using GPU
        solver = GPUSolver(A, tol=1e-8, maxiter=min(1000, N))
        G_cols = solver.solve(None, b)  # Shape: (N, m) on GPU
        
        # Extract blocks
        G_LR = G_cols[:m, :]
        G_RL = G_LR.conj().T
        
        # Broadening matrices (convert to GPU)
        GammaL = self._broadening(xp.asarray(SigmaL))
        GammaR = self._broadening(xp.asarray(SigmaR))
        
        # Fisher-Lee formula
        T = xp.real(xp.trace(GammaL @ G_LR @ GammaR @ G_RL))
        
        return float(max(cp.asnumpy(T), 0.0))
    
    def conductance(self, E: float) -> float:
        """
        Compute conductance at energy E.
        
        G = G0 · T(E)
        
        where G0 = 2e²/h ≈ 77.5 µS is the conductance quantum.
        
        Parameters
        ----------
        E : float
            Energy (eV)
            
        Returns
        -------
        G : float
            Conductance (S)
        """
        T = self.transmission(E)
        return G0_SI * T
    
    def IV(self, bias: float, mesh: EnergyMesh) -> Dict[str, np.ndarray]:
        """
        Compute current-voltage characteristics using Landauer formula.
        
        I(V) = (2e²/h) ∫ T(E) [f_L(E) - f_R(E)] dE
        
        where:
            μ_L = E_F + eV/2  (left contact)
            μ_R = E_F - eV/2  (right contact)
        
        Parameters
        ----------
        bias : float
            Bias voltage (V)
        mesh : EnergyMesh
            Energy grid for integration
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'E': Energy grid (eV)
            - 'T': Transmission values
            - 'fL', 'fR': Fermi distributions
            - 'I': Current (A)
            - 'muL', 'muR': Chemical potentials (eV)
        """
        # Chemical potentials
        muL = self.device.Ef + 0.5 * bias
        muR = self.device.Ef - 0.5 * bias
        
        # Compute transmission on energy grid
        grid = mesh.grid
        Tvals = np.empty_like(grid, dtype=float)
        
        print(f"Computing I-V at V={bias:.4f} V ({len(grid)} energy points)...")
        for i, E in enumerate(grid):
            Tvals[i] = self.transmission(E)
            if (i + 1) % max(1, len(grid) // 10) == 0:
                print(f"  Progress: {i+1}/{len(grid)}")
        
        # Fermi distributions (always on CPU for integration)
        fL = fermi(grid, muL, self.kT, gpu=False)
        fR = fermi(grid, muR, self.kT, gpu=False)
        
        # Landauer integral using trapezoidal rule
        integrand = Tvals * (fL - fR)
        dE = np.diff(grid) * E_CHARGE  # Convert eV to J
        integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dE)
        
        # Current in Amperes
        current = 2.0 * (E_CHARGE / H_PLANCK) * integral
        
        return {
            'E': grid,
            'T': Tvals,
            'fL': fL,
            'fR': fR,
            'I': current,
            'muL': muL,
            'muR': muR,
            'bias': bias
        }


# ============================================================================
# Utility Functions
# ============================================================================

def lift_to_spin(H_scalar: sp.csr_matrix) -> sp.csr_matrix:
    """
    Lift spinless Hamiltonian to spin-1/2 space.
    
    Creates 2×2 block structure: H_spin = I_2 ⊗ H_scalar
    
    Parameters
    ----------
    H_scalar : sparse matrix (N × N)
        Spinless Hamiltonian
        
    Returns
    -------
    H_spin : sparse matrix (2N × 2N)
        Spin-resolved Hamiltonian
    """
    I2 = sp.eye(2, format="csr")
    return sp.kron(I2, H_scalar, format="csr")


def make_1d_chain(n_cells: int = 50, t: float = 1.0, spin: bool = False) -> Device:
    """
    Create 1D tight-binding chain for testing.
    
    H = Σ_i ε_i c†_i c_i - t Σ_<i,j> (c†_i c_j + h.c.)
    
    Parameters
    ----------
    n_cells : int
        Number of sites in central region
    t : float
        Hopping parameter (eV)
    spin : bool
        Include spin degree of freedom (doubles system size)
        
    Returns
    -------
    device : Device
        1D chain device with semi-infinite leads
    """
    # Build scalar chain
    N = n_cells
    main_diag = np.zeros(N)
    off_diag = -t * np.ones(N - 1)
    H_scalar = sp.diags([off_diag, main_diag, off_diag], 
                        [-1, 0, 1], format="csr")
    
    # Lead unit cell (single site)
    H00_scalar = np.array([[0.0]])
    H01_scalar = np.array([[-t]])
    
    if spin:
        # Lift to spin space
        H = lift_to_spin(H_scalar)
        I2 = np.eye(2)
        H00 = np.kron(I2, H00_scalar)  # 2×2
        H01 = np.kron(I2, H01_scalar)  # 2×2
    else:
        H = H_scalar
        H00 = H00_scalar  # 1×1
        H01 = H01_scalar  # 1×1
    
    # Define leads
    left = PeriodicLead(H00=H00, H01=H01, tau_cpl=None)
    right = PeriodicLead(H00=H00, H01=H01, tau_cpl=None)
    
    return Device(H=H, S=None, left=left, right=right, Ef=0.0)


# ============================================================================
# Version and Info
# ============================================================================

__version__ = "2.0.0"
__author__ = "Custom KWANT Replacement"

def info():
    """Print library information."""
    import scipy
    print(f"ThoothQT Quantum Transport Library v{__version__}")
    print(f"GPU Support: {_CUPY_AVAILABLE}")
    if _CUPY_AVAILABLE:
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")


if __name__ == "__main__":
    print("=" * 70)
    print("ThoothQT - Quantum Transport Library")
    print("=" * 70)
    print()
    
    info()
    print()
    
    print("Running basic test...")
    print("-" * 70)
    
    # Create 1D chain
    device = make_1d_chain(n_cells=10, t=1.0, spin=False)
    print(f"Device: 1D chain, {device.H.shape[0]} sites")
    print(f"Lead size: {device.left.H00.shape[0]} orbitals")
    print()
    
    # Initialize engine
    engine = NEGFEngine(device, Temp=300, gpu=False)
    print()
    
    # Compute transmission at test energies...")
    test_energies = [-0.5, 0.0, 0.5, 1.0, 1.5]
    print(f"{'Energy (eV)':<15} {'Transmission':<15} {'Conductance (µS)':<20}")
    print("-" * 50)
    
    for E in test_energies:
        T = engine.transmission(E)
        G = engine.conductance(E)
        print(f"{E:<15.2f} {T:<15.6f} {G*1e6:<20.3f}")
    
    print()
    print("=" * 70)
    print("ThoothQT: Production-ready quantum transport for complex systems")
    print("=" * 70)
    print("✓ Test complete")
