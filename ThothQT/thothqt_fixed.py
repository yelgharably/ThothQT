"""
ThothQT: Quantum Transport and SCF-Compatible Modular Framework
---------------------------------------------------------------
Author: Youssef El Gharably (2025)
GPU-compatible NEGF framework replacing KWANT for modular, SCF-ready quantum sensing simulations.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time, os, json

# Optional GPU support ---------------------------------------------------
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspla
    xp = cp
    GPU_AVAILABLE = True
    print("[ThothQT] CuPy GPU backend detected. GPU acceleration enabled.")
except Exception:
    xp = np
    GPU_AVAILABLE = False
    print("[ThothQT] CuPy not found. Running on CPU.")

# ----------------------------- Utilities --------------------------------
def fermi(E, mu=0, T=300):
    kB = 8.617333262145e-5
    return 1.0 / (np.exp((E - mu) / (kB * T)) + 1)

def current_to_si(I_q):
    e = 1.602176634e-19
    h = 6.62607015e-34
    return I_q * (2 * e**2 / h)

def timer(func):
    def wrap(*a, **kw):
        t0 = time.time()
        res = func(*a, **kw)
        print(f"[Timer] {func.__name__}: {time.time()-t0:.3f}s")
        return res
    return wrap

# --------------------- Sancho-Rubio Decimator ---------------------------
class SanchoRubioDecimator:
    def __init__(self, H00, H01, max_iter=100, tol=1e-12):
        self.H00 = H00
        self.H01 = H01
        self.max_iter = max_iter
        self.tol = tol

    def surface_g(self, E):
        g = np.linalg.inv((E + 1j*1e-8)*np.eye(self.H00.shape[0]) - self.H00)
        alpha = self.H01.copy()
        beta = self.H01.conj().T
        for _ in range(self.max_iter):
            g_new = np.linalg.inv((E + 1j*1e-8)*np.eye(self.H00.shape[0]) - self.H00 - alpha @ g @ beta)
            if np.linalg.norm(g_new - g) < self.tol:
                return g_new
            g = g_new
        return g

# --------------------- Periodic Lead Class ------------------------------
class PeriodicLead:
    def __init__(self, H00, H01, tau_cpl=None, device_size=None, is_right=False):
        self.H00 = sp.csr_matrix(H00)
        self.H01 = sp.csr_matrix(H01)
        self.tau_cpl = sp.csr_matrix(tau_cpl) if tau_cpl is not None else None
        self.device_size = device_size  # Size of device Hamiltonian
        self.is_right = is_right  # Whether this is the right lead
        self.decimator = SanchoRubioDecimator(self.H00.toarray(), self.H01.toarray())

    def sigma(self, E, eta=1e-6):
        g_s = self.decimator.surface_g(E + 1j*eta)
        g_s = sp.csr_matrix(g_s)
        
        if self.tau_cpl is None:
            # For 1D chain: coupling is only to the first/last site
            device_size = self.device_size if self.device_size is not None else g_s.shape[0]
            
            # Create device-sized self-energy matrix
            sigma = sp.lil_matrix((device_size, device_size), dtype=complex)
            val = (self.H01[0,0]*self.H01[0,0].conj()) * g_s[0,0]
            
            if self.is_right:
                sigma[device_size-1, device_size-1] = val
            else:
                sigma[0, 0] = val
            
            return sigma.tocsr()
        else:
            # For systems with explicit coupling matrix
            sigma_lead = (self.tau_cpl @ g_s @ self.tau_cpl.getH())
            
            # Embed into device space if needed
            device_size = self.device_size if self.device_size is not None else sigma_lead.shape[0]
            
            if sigma_lead.shape[0] < device_size:
                sigma = sp.lil_matrix((device_size, device_size), dtype=complex)
                
                if self.is_right:
                    # Right lead couples to last sites
                    start_idx = device_size - sigma_lead.shape[0]
                    sigma[start_idx:, start_idx:] = sigma_lead.toarray()
                else:
                    # Left lead couples to first sites
                    sigma[:sigma_lead.shape[0], :sigma_lead.shape[0]] = sigma_lead.toarray()
                
                return sigma.tocsr()
            else:
                return sigma_lead.tocsr()

# ------------------------ NEGF Engine -----------------------------------
class NEGFEngine:
    def __init__(self, H, left_lead, right_lead, eta=1e-6, use_gpu=True):
        self.H = sp.csr_matrix(H)
        self.left = left_lead
        self.right = right_lead
        self.N = H.shape[0]
        self.eta = eta
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Set device size in leads if not already set
        if self.left.device_size is None:
            self.left.device_size = self.N
        if self.right.device_size is None:
            self.right.device_size = self.N

    def _to_backend(self, M):
        if self.use_gpu:
            return csp.csr_matrix(M)
        return M

    def _lin_solve(self, A):
        if self.use_gpu:
            I = csp.eye(A.shape[0])
            return cspla.cg(A, I)[0]
        else:
            return spla.inv(A)

    @timer
    def transmission(self, E):
        SigmaL = self.left.sigma(E, self.eta)
        SigmaR = self.right.sigma(E, self.eta)
        
        I = sp.identity(self.N, dtype=complex)
        A = (E + 1j*self.eta)*I - self.H - SigmaL - SigmaR
        G = spla.inv(A.tocsc())
        GammaL = 1j*(SigmaL - SigmaL.getH())
        GammaR = 1j*(SigmaR - SigmaR.getH())
        
        # Ensure all matrices are dense for trace calculation
        GammaL_dense = GammaL.toarray() if hasattr(GammaL, 'toarray') else GammaL
        GammaR_dense = GammaR.toarray() if hasattr(GammaR, 'toarray') else GammaR
        G_dense = G.toarray() if hasattr(G, 'toarray') else G
        
        # Debug: check magnitudes (only print occasionally)
        if abs(E) < 0.1:  # Only print near E=0
            print(f"Debug T @ E={E:.3f}: |Gamma_L|_max={np.max(np.abs(GammaL_dense)):.2e}, |Gamma_R|_max={np.max(np.abs(GammaR_dense)):.2e}")
            print(f"  |G|_max={np.max(np.abs(G_dense)):.2e}")
        
        T = np.real(np.trace(GammaL_dense @ G_dense @ GammaR_dense @ G_dense.conj().T))
        
        # Debug: print result
        if abs(E) < 0.1:
            print(f"  Transmission = {T:.6f}")
        
        return T

    @timer
    def DOS(self, E):
        SigmaL = self.left.sigma(E, self.eta)
        SigmaR = self.right.sigma(E, self.eta)
        I = sp.identity(self.N, dtype=complex)
        A = (E + 1j*self.eta)*I - self.H - SigmaL - SigmaR
        G = spla.inv(A.tocsc())
        
        # Ensure G is dense for trace calculation
        G_dense = G.toarray() if hasattr(G, 'toarray') else G
        return -np.imag(np.trace(G_dense)) / np.pi

# ------------------------- Graphene Builder -----------------------------
class GrapheneBuilder:
    def __init__(self, width=4, length=6, t=-2.7, edge_type='zigzag'):
        """
        Build graphene nanoribbon with proper honeycomb structure
        width: number of zigzag/armchair chains across
        length: number of unit cells along transport direction  
        t: nearest-neighbor hopping (typically -2.7 eV)
        edge_type: 'zigzag' or 'armchair'
        """
        self.width = width
        self.length = length  
        self.t = t
        self.edge_type = edge_type
        
        if edge_type == 'zigzag':
            self.H, self.H01, self.tau = self._build_zigzag_ribbon()
        else:
            self.H, self.H01, self.tau = self._build_armchair_ribbon()

    def _build_zigzag_ribbon(self):
        """Build zigzag graphene nanoribbon with proper honeycomb lattice"""
        
        # For zigzag ribbon: transport along x, width along y
        # Each unit cell has 2 atoms (A and B sublattices)  
        n_atoms_per_slice = self.width * 2  # 2 atoms per width unit
        n_total = n_atoms_per_slice * self.length
        
        H = sp.lil_matrix((n_total, n_total), dtype=complex)
        
        def get_atom_index(x, y, sublattice):
            """Get flat index for atom at (x,y) on sublattice A(0) or B(1)"""
            return x * n_atoms_per_slice + y * 2 + sublattice
        
        # Build honeycomb connections
        for x in range(self.length):
            for y in range(self.width):
                # Indices for A and B atoms in current unit cell
                a_idx = get_atom_index(x, y, 0)  # A sublattice
                b_idx = get_atom_index(x, y, 1)  # B sublattice
                
                # Intra-cell A-B hopping (vertical bonds in zigzag)
                if a_idx < n_total and b_idx < n_total:
                    H[a_idx, b_idx] = self.t
                
                # Inter-cell hoppings along transport direction (x)
                if x < self.length - 1:
                    # A atom connects to B atom in next cell
                    next_b = get_atom_index(x+1, y, 1)
                    if next_b < n_total:
                        H[a_idx, next_b] = self.t
                    
                    # B atom connects to A atom in next cell (with y-shift for zigzag)
                    if y > 0:  # Check boundary
                        next_a = get_atom_index(x+1, y-1, 0)  
                        if next_a < n_total:
                            H[b_idx, next_a] = self.t
        
        # Make Hamiltonian Hermitian
        H = H + H.getH()
        
        # Simplified lead coupling - minimal approach
        # Each lead couples to the edge atoms with single-atom coupling
        
        # For now, use simple single-site coupling (like 1D case)
        H01 = sp.csr_matrix([[self.t * 0.1]])  # Single weak coupling
        tau = H01.copy()
        
        # Note: This is a simplified approach. In a full implementation,
        # H01 would have the same structure as the device slice-to-slice coupling
        
        return H.tocsr(), H01, tau

    def _build_armchair_ribbon(self):
        """Build armchair graphene nanoribbon"""
        # Simplified armchair - use same structure but different orientation
        # For now, reuse zigzag logic (can be refined later)
        return self._build_zigzag_ribbon()

# --------------------- Validation Section -------------------------------
if __name__ == "__main__":
    os.makedirs("validation_results", exist_ok=True)

    # 1D chain validation -----------------------------------------------
    print("\n[Validation] 1D chain test")
    N = 20
    t = -1.0
    H = sp.diags([t*np.ones(N-1), np.zeros(N), t*np.ones(N-1)], [-1, 0, 1])
    H00 = sp.csr_matrix([[0]])
    H01 = sp.csr_matrix([[t]])
    tau = H01.copy()

    left = PeriodicLead(H00, H01, tau, device_size=N, is_right=False)
    right = PeriodicLead(H00, H01, tau, device_size=N, is_right=True)

    engine = NEGFEngine(H, left, right, use_gpu=True)
    energies = np.linspace(-3, 3, 200)
    Tvals = [engine.transmission(E) for E in energies]
    DOSvals = [engine.DOS(E) for E in energies]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(energies, Tvals)
    plt.title("1D Chain Transmission")
    plt.subplot(1,2,2)
    plt.plot(energies, DOSvals)
    plt.title("1D Chain DOS")
    plt.tight_layout()
    plt.savefig("validation_results/1d_chain_validation.png", dpi=300)

    # Graphene nanoribbon test ------------------------------------------
    print("\n[Validation] Graphene nanoribbon test")
    
    # Create proper zigzag graphene nanoribbon
    gb = GrapheneBuilder(width=3, length=4, t=-2.7, edge_type='zigzag')
    print(f"Graphene nanoribbon: {gb.H.shape[0]} atoms")
    
    left = PeriodicLead(gb.H01, gb.H01, gb.tau, is_right=False)
    right = PeriodicLead(gb.H01, gb.H01, gb.tau, is_right=True)
    engine2 = NEGFEngine(gb.H, left, right, use_gpu=True)
    
    # Use smaller energy range for better resolution
    energies = np.linspace(-4, 4, 100)  
    Tvals2 = [engine2.transmission(E) for E in energies]
    
    plt.figure(figsize=(10,6))
    
    # Main transmission plot
    plt.subplot(1, 2, 1)
    plt.plot(energies, Tvals2, 'b-', linewidth=2, label="Zigzag Graphene T(E)")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Transmission")
    plt.title("Graphene Nanoribbon Transmission")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)  # Ensure y starts at 0
    
    # Log plot to see small features
    plt.subplot(1, 2, 2)  
    plt.semilogy(energies, np.array(Tvals2) + 1e-10, 'g-', linewidth=2)
    plt.xlabel("Energy (eV)")
    plt.ylabel("log(Transmission)")
    plt.title("Transmission (Log Scale)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("validation_results/graphene_validation.png", dpi=300)

    print("\nValidation complete. Results saved in 'validation_results/'.")
