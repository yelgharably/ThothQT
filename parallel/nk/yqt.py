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

def fermi(E,mu,kT,gpu=False):
    """Fermi-Dirac distribution function."""
    x = (E-mu)/(max(kT,1e-12))
    if gpu and _CUPY_OK:
        out = cp.empty_like(x, dtype=float)
        pos = x > 50; neg = x <-50; mid = ~(pos | neg)
        out[pos] = 0.0; out[neg] = 1.0; out[mid] = 1.0/(1.0 + cp.exp(x[mid]))
        return out
    else:
        out = np.empty_like(x, dtype=float)
        pos = x > 50; neg = x <-50; mid = ~(pos | neg)
        out[pos] = 0.0; out[neg] = 1.0; out[mid] = 1.0/(1.0 + np.exp(x[mid]))
        return out
    
@dataclass
class PeriodicLead:
    H00: Union[np.ndarray, sp.spmatrix]
    H01: Union[np.ndarray, sp.spmatrix]
    tau_cpl: Optional[Union[np.ndarray, sp.spmatrix]] = None

class SanchoRubioDecimator:
    def __init__(self, lead: PeriodicLead, eta: float=1e-6,
                  max_iter: int=1000, tol: float=1e-12, gpu: bool=False):
        self.lead = lead
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.gpu = gpu

        if self.gpu and not _CUPY_OK:
            warnings.warn("GPU Requested but CUPY unavailable. Falling back to CPU.")
            self.gpu = False
        
        self.xp = cp if self.gpu else np
        self.sp = cpx_sp if self.gpu else sp

        self.H00 = self._to_array(self.lead.H00)
        self.H01 = self._to_array(self.lead.H01)
        self.tau_cpl = self._to_array(self.lead.tau_cpl) if self.lead.tau_cpl is not None else None

    def _to_backend(self,arr):
        return self.xp.asarray(arr)
    
    def _inv(self,arr):
        return self.xp.linalg.inv(arr)
    
    def surface_g(self,E:float):
        xp = self.xp
        H00, H01 = self.H00, self.H01
        zI = (E + 1j*self.eta)*xp.eye(H00.shape[0], dtype=complex)
        a = H00.copy()
        alpha = H01.copy()
        beta = H01.conj().T.copy()
        g = self._inv(zI - a)
        for _ in range(self.max_iter):
            a_new = a + alpha @ g @ beta
            alpha_new = alpha @ g @ alpha
            beta_new = beta @ g @ beta
            g_new = self._inv(zI - a_new)
            if xp.linalg.norm(g_new - g, ord='fro') < self.tol:
                return g_new
            a, alpha, beta, g = a_new, alpha_new, beta_new, g_new
        return g
    
@dataclass
class Device:
    H: sp.csr_matrix
    S: sp.csr_matrix | None
    left: PeriodicLead
    right: PeriodicLead
    Ef: float = 0.0

class LinearSolver:
    def solve(self, A, b):
        raise NotImplementedError("LinearSolver is an abstract base class.")
    
class CPUSolver(LinearSolver):
    def __init__(self,A_csr:sp.csr_matrix):
        self.lu = spla.splu(A_csr.tocsc())

    def solve(self,A_unused,B_np:np.ndarray):
        return self.lu.solve(B_np)
    
class GPUSolver(LinearSolver):
    def __init__(self,A_csr, tol = 1e-9, maxiter=1000):
        if not _CUPY_OK:
            raise RuntimeError("CUPY is not available for GPU computations.")
        self.A = A_csr
        self.tol = tol
        self.maxiter = maxiter

        D = self.A.diagonal()
        D = cp.where(cp.abs(D) > 1e-30, D, 1.0+0j)
        self.Dinv = 1.0/D

    def _apply_prec(self,r):
        return self.Dinv * r
    
    def solve(self,A_unused,B_cp):
        n,m = B_cp.shape
        X = cp.zeros((n,m), dtype=B_cp.dtype)
        Aop = cpx_spla.LinearOperator(self.A.shape, matvec=lambda x: self.A.dot(x))

        def M_inv(x):
            return self._apply_prec(x)

        M = cpx_spla.LinearOperator(self.A.shape, matvec=M_inv)
        for j in range(m):
            bj = B_cp[:,j]
            yj, info = cpx_spla.gmres(Aop, bj, M=M, tol=self.tol, maxiter=self.maxiter)
            if info != 0:
                raise warnings.warn(f"GMRES did not converge for column {j}, info={info}")
            X[:,j] = M.matvec(yj)
        return X
    
class EnergyMesh:
    def __init__(self,Emin:float, Emax: float, n:int, refine_at: Tuple[float,...]=(),refine_pts: int = 200):
        base = np.linspace(Emin,Emax,n)
        extra = []
        for E0 in refine_at:
            extra.append(np.linspace(max(Emin,E0-0.1), min(Emax,E0+0.1), refine_pts))
        self.grid = np.unique(np.concatenate([base]+extra)) if extra else base

class NEGFEngine:
    def __init__(self,device: Device,Temp, eta: float=1e-6, gpu: bool=False):
        self.device = device
        self.kT = Temp * 8.617333262145e-5
        self.eta = eta
        self.gpu = gpu and _CUPY_OK
        self.xp = cp if self.gpu else np
        self.spmod = cpx_sp if self.gpu else sp  # Fix: assign correct sparse module

        self.decL = SanchoRubioDecimator(device.left, eta=eta, gpu=self.gpu)
        self.decR = SanchoRubioDecimator(device.right, eta=eta, gpu=self.gpu)

    def _assemble_A(self, E: float, SigmaL, SigmaR):
        H = self.device.H
        N = H.shape[0]
        if self.backend == "cpu":
            if self.device.S is None:
                A = ((E + 1j*self.eta) * sp.eye(N, format="csr")) - H
            else:
                A = (E + 1j*self.eta) * self.device.S - H
            # apply sigmas to first/last m blocks
            m = self.device.left.H00.shape[0]
            A = A.tolil()
            SigmaL_sparse = sp.csr_matrix(SigmaL)
            SigmaR_sparse = sp.csr_matrix(SigmaR)
            A[:m, :m] -= SigmaL_sparse
            A[-m:, -m:] -= SigmaR_sparse
            SigmaL_sparse = SigmaL
            if not sp.isspmatrix(SigmaR):
                SigmaR_sparse = sp.csr_matrix(SigmaR)
            else:
                SigmaR_sparse = SigmaR
            A[:m, :m] -= SigmaL_sparse
            A[-m:, -m:] -= SigmaR_sparse
            return A.tocsr()

        # GPU path
        xp, spm = self.xp, self.spmod
        if self.device.S is None:
            A = ((E + 1j*self.eta) * spm.identity(N, dtype=complex, format="csr")) - spm.csr_matrix(self.device.H)
        else:
            A = (E + 1j*self.eta) * spm.csr_matrix(self.device.S) - spm.csr_matrix(self.device.H)
        m = self.device.left.H00.shape[0]
        # Convert SigmaL/R to GPU and add to diagonal blocks
        SL = xp.asarray(SigmaL)
        SR = xp.asarray(SigmaR)
        # Build block updates as sparse:
        # Create sparse matrices for block updates
        SL_block = spm.csr_matrix((SL, (xp.arange(m), xp.arange(m))), shape=(N, N))
        SR_block = spm.csr_matrix((SR, (xp.arange(N-m, N), xp.arange(N-m, N))), shape=(N, N))
        A = A - SL_block - SR_block
        return A

    @staticmethod
    def _gamma(Sigma):
        return 1j * (Sigma - Sigma.conj().T)

    def transmission(self, E: float) -> float:
        SigmaL = self.decL.sigma(E)
        SigmaR = self.decR.sigma(E)
        A = self._assemble_A(E, SigmaL, SigmaR)

        m = self.dev.left.H0.shape[0]
        N = A.shape[0]

        if self.backend == "cpu":
            lu = spla.splu(A.tocsc())
            # RHS: unit columns on right-contact subspace
            B = np.zeros((N, m), dtype=complex)
            B[N-m:N, :] = np.eye(m, dtype=complex)
            Gr_cols = lu.solve(B)  # N x m
            idxL = np.arange(0, m)
            idxR = np.arange(N-m, N)
            GL = self._gamma(SigmaL)
            GR = self._gamma(SigmaR)
            Gr_LR = Gr_cols[idxL, :]      # m x m
            Ga_RL = Gr_LR.conj().T
            T = np.real(np.trace(GL @ Gr_LR @ GR @ Ga_RL))
            return float(max(T, 0.0))

            # GPU path (GMRES)
            xp = self.xp
            spm = self.spmod
            # Build RHS on GPU
            B = xp.zeros((N, m), dtype=complex)
            B[N-m:N, :] = xp.eye(m, dtype=complex)
            # Solve
            solver = GpuGMRESSolver(A_csr_gpu=A.tocsr(), tol=1e-8, maxiter=None)
            Gr_cols = solver.solve(None, B)  # N x m (gpu)
            idxL = xp.arange(0, m)
            idxR = xp.arange(N-m, N)
            GL = self._gamma(SigmaL)
            GR = self._gamma(SigmaR)
            Gr_LR = Gr_cols[idxL, :]               # m x m
            Ga_RL = Gr_LR.conj().T
            T = xp.real(xp.trace(GL @ Gr_LR @ GR @ Ga_RL))
            return float(max(cp.asnumpy(T), 0.0))

        def IV(self, bias: float, mesh: EnergyMesh) -> Dict[str, np.ndarray]:
            muL = self.dev.Ef + 0.5 * bias
            muR = self.dev.Ef - 0.5 * bias

            grid = mesh.grid
            Tvals = np.empty_like(grid, dtype=float)
            for i, E in enumerate(grid):
                Tvals[i] = self.transmission(E)

            # Fermi terms (CPU)
            fL = _fermi_numpy(grid, muL, self.kT)
            fR = _fermi_numpy(grid, muR, self.kT)

            integrand = Tvals * (fL - fR)
            e_C = 1.602176634e-19
            dE = np.diff(grid) * e_C
            integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dE)
            current_A = 2.0 * (e_C / 6.62607015e-34) * integral
            return {"E": grid, "T": Tvals, "I": current_A, "muL": muL, "muR": muR}
        

def lift_to_spin(H_scalar: sp.csr_matrix):
    I2 = sp.eye(2, format="csr")
    return sp.kron(I2, H_scalar, format="csr")          # 2x2 block per site

def make_1d_chain_spin(n_cells=50, t=1.0):
    # scalar chain
    N = n_cells
    main = np.zeros(N)
    off  = -t*np.ones(N-1)
    Hs   = sp.diags([off, main, off], [-1,0,1], format="csr")

    # lift device and leads to 2x2 blocks
    H    = lift_to_spin(Hs)
    H0s  = np.array([[0.0]])
    H1s  = np.array([[-t]])
    I2   = np.eye(2)
    H0   = np.kron(I2, H0s)      # 2x2
    H1   = np.kron(I2, H1s)      # 2x2
    tau  = np.kron(I2, np.array([[-t]]))  # 2x2

    left  = PeriodicLead(H0=H0, H1=H1, tau_cpl=tau)
    right = PeriodicLead(H0=H0, H1=H1, tau_cpl=tau)
    return Device(H=H, S=None, left=left, right=right, Ef=0.0)