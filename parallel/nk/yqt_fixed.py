"""
Fixed YQT Implementation

Corrected version of yqt.py with all critical issues resolved.
"""

from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    import cupyx.scipy.sparse.linalg as cpx_spla
    _CUPY_OK = (cp.cuda.runtime.getDeviceCount() > 0)
except Exception:
    cp, cpx_sp, cpx_spla = None, None, None
    _CUPY_OK = False

def fermi(E, mu, kT, gpu=False):
    """Fermi-Dirac distribution function with overflow protection."""
    x = (E - mu) / max(kT, 1e-12)
    if gpu and _CUPY_OK:
        out = cp.empty_like(x, dtype=float)
        pos = x > 50
        neg = x < -50
        mid = ~(pos | neg)
        out[pos] = 0.0
        out[neg] = 1.0
        out[mid] = 1.0 / (1.0 + cp.exp(x[mid]))
        return out
    else:
        out = np.empty_like(x, dtype=float)
        pos = x > 50
        neg = x < -50
        mid = ~(pos | neg)
        out[pos] = 0.0
        out[neg] = 1.0
        out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
        return out


@dataclass
class PeriodicLead:
    """Semi-infinite periodic lead definition."""
    H00: Union[np.ndarray, sp.spmatrix]  # On-site Hamiltonian of unit cell
    H01: Union[np.ndarray, sp.spmatrix]  # Hopping to next unit cell
    tau_cpl: Optional[Union[np.ndarray, sp.spmatrix]] = None  # Coupling to device


class SanchoRubioDecimator:
    """
    Sancho-Rubio iterative decimation for surface Green's function.
    Computes g_s = [E*I - H00 - H01*g_s*H01†]^{-1}
    """
    
    def __init__(self, lead: PeriodicLead, eta: float = 1e-6,
                 max_iter: int = 1000, tol: float = 1e-12, gpu: bool = False):
        self.lead = lead
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.gpu = gpu

        if self.gpu and not _CUPY_OK:
            warnings.warn("GPU requested but CuPy unavailable. Falling back to CPU.")
            self.gpu = False
        
        self.xp = cp if self.gpu else np
        self.sp = cpx_sp if self.gpu else sp

        # Convert to dense arrays on appropriate backend
        self.H00 = self._to_array(self.lead.H00)
        self.H01 = self._to_array(self.lead.H01)
        self.tau_cpl = self._to_array(self.lead.tau_cpl) if self.lead.tau_cpl is not None else None

    def _to_array(self, arr):
        """Convert input to dense array on current backend."""
        if arr is None:
            return None
        
        # Convert sparse to dense if needed
        if sp.issparse(arr):
            arr = arr.toarray()
        
        # Convert to backend (GPU/CPU)
        if self.gpu:
            return cp.asarray(arr)
        else:
            return np.asarray(arr)
    
    def _inv(self, arr):
        """Matrix inversion on current backend."""
        return self.xp.linalg.inv(arr)
    
    def surface_g(self, E: float):
        """
        Compute surface Green's function using Sancho-Rubio decimation.
        Returns: g_s(E) for semi-infinite lead
        """
        xp = self.xp
        H00, H01 = self.H00, self.H01
        zI = (E + 1j * self.eta) * xp.eye(H00.shape[0], dtype=complex)
        
        # Initial values
        a = H00.copy()
        alpha = H01.copy()
        beta = H01.conj().T.copy()
        g = self._inv(zI - a)
        
        for iteration in range(self.max_iter):
            # Decimation step
            a_new = a + alpha @ g @ beta
            alpha_new = alpha @ g @ alpha
            beta_new = beta @ g @ beta
            g_new = self._inv(zI - a_new)
            
            # Check convergence
            if xp.linalg.norm(g_new - g, ord='fro') < self.tol:
                return g_new
            
            a, alpha, beta, g = a_new, alpha_new, beta_new, g_new
        
        warnings.warn(f"Sancho-Rubio did not converge after {self.max_iter} iterations")
        return g
    
    def sigma(self, E: float):
        """
        Compute self-energy: Sigma = tau† × g_surface × tau
        """
        g_s = self.surface_g(E)
        
        if self.tau_cpl is None:
            # If no explicit coupling, assume H01 is the coupling
            tau = self.H01
        else:
            tau = self.tau_cpl
        
        # Sigma = tau† × g_s × tau
        Sigma = tau.conj().T @ g_s @ tau
        return Sigma


@dataclass
class Device:
    """Central device region with left and right leads."""
    H: sp.csr_matrix              # Device Hamiltonian
    S: sp.csr_matrix | None       # Overlap matrix (None for orthogonal basis)
    left: PeriodicLead            # Left lead
    right: PeriodicLead           # Right lead
    Ef: float = 0.0               # Fermi level


class LinearSolver:
    """Abstract base class for linear system solvers."""
    def solve(self, A, b):
        raise NotImplementedError("LinearSolver is an abstract base class.")


class CPUSolver(LinearSolver):
    """CPU sparse direct solver using SuperLU."""
    def __init__(self, A_csr: sp.csr_matrix):
        self.lu = spla.splu(A_csr.tocsc())

    def solve(self, A_unused, B_np: np.ndarray):
        return self.lu.solve(B_np)


class GPUSolver(LinearSolver):
    """GPU iterative solver using GMRES with diagonal preconditioner."""
    def __init__(self, A_csr, tol=1e-9, maxiter=1000):
        if not _CUPY_OK:
            raise RuntimeError("CuPy is not available for GPU computations.")
        self.A = A_csr
        self.tol = tol
        self.maxiter = maxiter

        # Diagonal preconditioner
        D = self.A.diagonal()
        D = cp.where(cp.abs(D) > 1e-30, D, 1.0 + 0j)
        self.Dinv = 1.0 / D

    def _apply_prec(self, r):
        """Apply diagonal preconditioner."""
        return self.Dinv * r
    
    def solve(self, A_unused, B_cp):
        """Solve A × X = B using GMRES with preconditioning."""
        n, m = B_cp.shape
        X = cp.zeros((n, m), dtype=B_cp.dtype)
        Aop = cpx_spla.LinearOperator(self.A.shape, matvec=lambda x: self.A.dot(x))

        def M_inv(x):
            return self._apply_prec(x)

        M = cpx_spla.LinearOperator(self.A.shape, matvec=M_inv)
        
        for j in range(m):
            bj = B_cp[:, j]
            yj, info = cpx_spla.gmres(Aop, bj, M=M, tol=self.tol, maxiter=self.maxiter)
            if info != 0:
                warnings.warn(f"GMRES did not converge for column {j}, info={info}")
            X[:, j] = yj  # FIX: Return yj directly, not M.matvec(yj)
        return X


class EnergyMesh:
    """Energy grid with optional refinement around specific points."""
    def __init__(self, Emin: float, Emax: float, n: int,
                 refine_at: Tuple[float, ...] = (), refine_pts: int = 200):
        base = np.linspace(Emin, Emax, n)
        extra = []
        for E0 in refine_at:
            extra.append(np.linspace(max(Emin, E0 - 0.1), min(Emax, E0 + 0.1), refine_pts))
        self.grid = np.unique(np.concatenate([base] + extra)) if extra else base


class NEGFEngine:
    """
    Non-Equilibrium Green's Function engine for quantum transport.
    """
    
    def __init__(self, device: Device, Temp, eta: float = 1e-6, gpu: bool = False):
        self.device = device
        self.kT = Temp * 8.617333262145e-5  # Convert K to eV
        self.eta = eta
        self.gpu = gpu and _CUPY_OK
        self.backend = "gpu" if self.gpu else "cpu"  # FIX: Set backend attribute
        
        self.xp = cp if self.gpu else np
        self.spmod = cpx_sp if self.gpu else sp

        # Initialize lead decimators
        self.decL = SanchoRubioDecimator(device.left, eta=eta, gpu=self.gpu)
        self.decR = SanchoRubioDecimator(device.right, eta=eta, gpu=self.gpu)

    def _assemble_A(self, E: float, SigmaL, SigmaR):
        """
        Assemble system matrix: A = E*S - H - Sigma_L - Sigma_R
        """
        H = self.device.H
        N = H.shape[0]
        m = self.device.left.H00.shape[0]  # Lead block size
        
        if self.backend == "cpu":
            # CPU path with sparse matrices
            if self.device.S is None:
                A = ((E + 1j * self.eta) * sp.eye(N, format="csr")) - H
            else:
                A = (E + 1j * self.eta) * self.device.S - H
            
            # Add self-energies to corner blocks
            A = A.tolil()  # Convert to lil for efficient block modification
            A[:m, :m] -= sp.csr_matrix(SigmaL)
            A[-m:, -m:] -= sp.csr_matrix(SigmaR)
            return A.tocsr()

        # GPU path
        xp, spm = self.xp, self.spmod
        if self.device.S is None:
            A = ((E + 1j * self.eta) * spm.identity(N, dtype=complex, format="csr")) - spm.csr_matrix(self.device.H)
        else:
            A = (E + 1j * self.eta) * spm.csr_matrix(self.device.S) - spm.csr_matrix(self.device.H)
        
        # Convert to lil for block updates
        A = A.tolil()
        SL_gpu = xp.asarray(SigmaL)
        SR_gpu = xp.asarray(SigmaR)
        A[:m, :m] -= spm.csr_matrix(SL_gpu)
        A[-m:, -m:] -= spm.csr_matrix(SR_gpu)
        return A.tocsr()

    @staticmethod
    def _gamma(Sigma):
        """Broadening matrix: Gamma = i(Sigma - Sigma†)"""
        return 1j * (Sigma - Sigma.conj().T)

    def transmission(self, E: float) -> float:
        """
        Compute transmission at energy E using Fisher-Lee formula:
        T = Tr[Gamma_L × G × Gamma_R × G†]
        """
        # Get self-energies from leads
        SigmaL = self.decL.sigma(E)
        SigmaR = self.decR.sigma(E)
        
        # Assemble system matrix
        A = self._assemble_A(E, SigmaL, SigmaR)
        
        m = self.device.left.H00.shape[0]  # FIX: Use H00 not H0
        N = A.shape[0]

        if self.backend == "cpu":
            # CPU: Direct solver
            lu = spla.splu(A.tocsc())
            # RHS: unit columns on right-contact subspace
            B = np.zeros((N, m), dtype=complex)
            B[N - m:N, :] = np.eye(m, dtype=complex)
            Gr_cols = lu.solve(B)  # N × m
            
            # Extract relevant blocks
            GL = self._gamma(SigmaL)
            GR = self._gamma(SigmaR)
            Gr_LR = Gr_cols[:m, :]  # Top m rows
            Ga_RL = Gr_LR.conj().T
            
            # Fisher-Lee formula
            T = np.real(np.trace(GL @ Gr_LR @ GR @ Ga_RL))
            return float(max(T, 0.0))
        
        # GPU path (GMRES)
        xp = self.xp
        # Build RHS on GPU
        B = xp.zeros((N, m), dtype=complex)
        B[N - m:N, :] = xp.eye(m, dtype=complex)
        
        # Solve using GMRES
        solver = GPUSolver(A, tol=1e-8, maxiter=None)
        Gr_cols = solver.solve(None, B)  # N × m (gpu)
        
        # Extract and compute transmission
        GL = self._gamma(SigmaL)
        GR = self._gamma(SigmaR)
        Gr_LR = Gr_cols[:m, :]
        Ga_RL = Gr_LR.conj().T
        T = xp.real(xp.trace(GL @ Gr_LR @ GR @ Ga_RL))
        return float(max(cp.asnumpy(T), 0.0))

    def IV(self, bias: float, mesh: EnergyMesh) -> Dict[str, np.ndarray]:  # FIX: Un-indented to class level
        """
        Compute current-voltage characteristics using Landauer formula:
        I = (2e²/h) ∫ T(E) [f_L(E) - f_R(E)] dE
        """
        muL = self.device.Ef + 0.5 * bias
        muR = self.device.Ef - 0.5 * bias

        grid = mesh.grid
        Tvals = np.empty_like(grid, dtype=float)
        for i, E in enumerate(grid):
            Tvals[i] = self.transmission(E)

        # Fermi distributions (always on CPU for integration)
        fL = fermi(grid, muL, self.kT, gpu=False)  # FIX: Use fermi not _fermi_numpy
        fR = fermi(grid, muR, self.kT, gpu=False)

        # Trapezoidal integration
        integrand = Tvals * (fL - fR)
        e_C = 1.602176634e-19
        dE = np.diff(grid) * e_C
        integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dE)
        current_A = 2.0 * (e_C / 6.62607015e-34) * integral
        
        return {"E": grid, "T": Tvals, "I": current_A, "muL": muL, "muR": muR}


# Utility functions
def lift_to_spin(H_scalar: sp.csr_matrix):
    """Lift spinless Hamiltonian to spin-1/2 (2×2 blocks per site)."""
    I2 = sp.eye(2, format="csr")
    return sp.kron(I2, H_scalar, format="csr")


def make_1d_chain_spin(n_cells=50, t=1.0):
    """
    Create 1D tight-binding chain with spin degree of freedom.
    """
    # Scalar chain
    N = n_cells
    main = np.zeros(N)
    off = -t * np.ones(N - 1)
    Hs = sp.diags([off, main, off], [-1, 0, 1], format="csr")

    # Lift device and leads to 2×2 spin blocks
    H = lift_to_spin(Hs)
    
    # Lead Hamiltonians
    H0s = np.array([[0.0]])
    H1s = np.array([[-t]])
    I2 = np.eye(2)
    H0 = np.kron(I2, H0s)  # 2×2
    H1 = np.kron(I2, H1s)  # 2×2
    tau = np.kron(I2, np.array([[-t]]))  # 2×2

    # FIX: Use H00 and H01 parameter names
    left = PeriodicLead(H00=H0, H01=H1, tau_cpl=tau)
    right = PeriodicLead(H00=H0, H01=H1, tau_cpl=tau)
    
    return Device(H=H, S=None, left=left, right=right, Ef=0.0)


if __name__ == "__main__":
    print("YQT - Fixed Implementation")
    print("=" * 60)
    print()
    print("Testing 1D chain...")
    
    device = make_1d_chain_spin(n_cells=10, t=1.0)
    engine = NEGFEngine(device, Temp=300, gpu=False)
    
    print(f"Device size: {device.H.shape[0]} sites")
    print(f"Lead block size: {device.left.H00.shape[0]}")
    print()
    
    # Test transmission at a few energies
    test_energies = [0.0, 0.5, 1.0, 1.5]
    print("Transmission:")
    for E in test_energies:
        T = engine.transmission(E)
        print(f"  E = {E:+.2f} eV  →  T = {T:.6f}")
    
    print()
    print("✓ YQT fixed version test complete")
