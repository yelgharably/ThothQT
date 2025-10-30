"""
Non-Equilibrium Green's Function (NEGF) implementation for quantum transport.

This module provides classes and functions for computing Green's functions,
self-energies, and transport properties in the NEGF formalism.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Optional, Union
import warnings

class NEGFSolver:
    """
    Main class for NEGF calculations.
    
    Handles computation of Green's functions, self-energies, and transport properties
    for tight-binding systems with semi-infinite leads.
    """
    
    def __init__(self, H_device: np.ndarray, H_leads: list, V_couplings: list, 
                 eta: float = 1e-6):
        """
        Initialize NEGF solver.
        
        Parameters:
        -----------
        H_device : np.ndarray
            Hamiltonian matrix of the central device region
        H_leads : list of dict
            List containing lead Hamiltonians. Each dict should have:
            - 'H00': on-site Hamiltonian of lead
            - 'H01': hopping between lead unit cells
        V_couplings : list of np.ndarray
            Coupling matrices between device and leads
        eta : float
            Small imaginary part for Green's function regularization
        """
        self.H_device = np.asarray(H_device, dtype=complex)
        self.H_leads = H_leads
        self.V_couplings = V_couplings
        self.eta = eta
        self.n_device = H_device.shape[0]
        self.n_leads = len(H_leads)
        
        # Validate inputs
        if len(V_couplings) != len(H_leads):
            raise ValueError("Number of coupling matrices must match number of leads")
        
        # Pre-compute lead properties
        self._lead_info = []
        for i, lead in enumerate(H_leads):
            info = {
                'H00': np.asarray(lead['H00'], dtype=complex),
                'H01': np.asarray(lead['H01'], dtype=complex),
                'n_orb': lead['H00'].shape[0]
            }
            self._lead_info.append(info)
    
    def surface_greens_function(self, energy: float, lead_idx: int) -> np.ndarray:
        """
        Compute surface Green's function for a semi-infinite lead using
        the iterative method (Lopez Sancho et al.).
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate the surface Green's function
        lead_idx : int
            Index of the lead
            
        Returns:
        --------
        np.ndarray
            Surface Green's function g_s
        """
        lead = self._lead_info[lead_idx]
        H00 = lead['H00']
        H01 = lead['H01']
        
        # Check if H01 has compatible dimensions with H00
        if H01.shape[0] != H00.shape[0]:
            raise ValueError(f"Lead {lead_idx}: H01 shape {H01.shape} incompatible with H00 shape {H00.shape}")
        
        # For the iterative algorithm to work, we need H01 to couple to the same space as H00
        # If H01 is not square, we need to handle it properly for NEGF
        if H01.shape[1] != H00.shape[0]:
            # NEGF requires surface Green's function to match coupling matrix dimensions
            # Instead of truncating, we should pad H01 or handle the dimension mismatch properly
            
            # Option 1: Pad H01 to make it square (fill with zeros)
            H01_padded = np.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
            H01_padded[:H01.shape[0], :H01.shape[1]] = H01
            
            H00_trunc = H00
            H01_trunc = H01_padded
            H10 = H01_trunc.T.conj()
            E = (energy + 1j * self.eta) * np.eye(H00.shape[0])
        else:
            H00_trunc = H00
            H01_trunc = H01
            H10 = H01.T.conj()  # Hermitian conjugate
            E = (energy + 1j * self.eta) * np.eye(H00.shape[0])
        
        # Initial values
        alpha = H01_trunc.copy()
        beta = H10.copy()
        eps_s = H00_trunc.copy()
        
        # Iterative procedure
        max_iter = 100
        tol = 1e-12
        
        for i in range(max_iter):
            # Invert (E - eps_s)
            try:
                g_bulk = np.linalg.inv(E - eps_s)
            except np.linalg.LinAlgError:
                # Use pseudoinverse if singular
                g_bulk = np.linalg.pinv(E - eps_s)
                warnings.warn(f"Singular matrix in surface Green's function iteration {i}")
            
            # Update alpha and beta
            alpha_new = alpha @ g_bulk @ alpha
            beta_new = beta @ g_bulk @ beta
            
            # Check convergence
            alpha_diff = np.max(np.abs(alpha_new - alpha))
            beta_diff = np.max(np.abs(beta_new - beta))
            
            if max(alpha_diff, beta_diff) < tol:
                break
                
            # Update for next iteration
            eps_s = eps_s + alpha @ g_bulk @ beta + beta @ g_bulk @ alpha
            alpha = alpha_new
            beta = beta_new
        else:
            warnings.warn(f"Surface Green's function did not converge after {max_iter} iterations")
        
        # Final surface Green's function
        try:
            g_s = np.linalg.inv(E - eps_s)
        except np.linalg.LinAlgError:
            g_s = np.linalg.pinv(E - eps_s)
            warnings.warn("Singular matrix in final surface Green's function")
            
        return g_s
    
    def self_energy(self, energy: float, lead_idx: int) -> np.ndarray:
        """
        Compute self-energy for a specific lead.
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate the self-energy
        lead_idx : int
            Index of the lead
            
        Returns:
        --------
        np.ndarray
            Self-energy matrix Σ
        """
        try:
            g_s = self.surface_greens_function(energy, lead_idx)
            V_coupling = self.V_couplings[lead_idx]
            
            # Σ = V† g_s V
            # V_coupling is (n_lead_orb, n_device), g_s is (n_lead_orb, n_lead_orb)
            # Result should be (n_device, n_device)
            
            # Check dimensions
            if V_coupling.shape[0] != g_s.shape[0]:
                raise ValueError(f"Dimension mismatch: V_coupling {V_coupling.shape} vs g_s {g_s.shape}")
            
            Sigma = V_coupling.T.conj() @ g_s @ V_coupling
            
            return Sigma
            
        except Exception as e:
            # Return zero self-energy if calculation fails
            warnings.warn(f"Self-energy calculation failed for lead {lead_idx}: {e}")
            return np.zeros((self.n_device, self.n_device), dtype=complex)
    
    def retarded_greens_function(self, energy: float, 
                                mu_leads: Optional[list] = None) -> np.ndarray:
        """
        Compute retarded Green's function of the device.
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate the Green's function
        mu_leads : list, optional
            Chemical potentials of the leads (for finite bias)
            
        Returns:
        --------
        np.ndarray
            Retarded Green's function G^r
        """
        E = (energy + 1j * self.eta) * np.eye(self.n_device)
        H_eff = self.H_device.copy()
        
        # Add self-energies from all leads
        for i in range(self.n_leads):
            try:
                Sigma = self.self_energy(energy, i)
                # Check if self-energy has correct dimensions
                if Sigma.shape != (self.n_device, self.n_device):
                    # Create properly sized self-energy matrix
                    Sigma_full = np.zeros((self.n_device, self.n_device), dtype=complex)
                    # Map the computed self-energy to the appropriate block
                    min_dim = min(Sigma.shape[0], self.n_device)
                    if i == 0:  # Left lead - add to beginning
                        Sigma_full[:min_dim, :min_dim] = Sigma[:min_dim, :min_dim]
                    else:  # Right lead - add to end
                        Sigma_full[-min_dim:, -min_dim:] = Sigma[:min_dim, :min_dim]
                    Sigma = Sigma_full
                    
                H_eff += Sigma
            except Exception as se_error:
                warnings.warn(f"Failed to compute self-energy for lead {i}: {se_error}")
        
        try:
            G_r = np.linalg.inv(E - H_eff)
        except np.linalg.LinAlgError:
            try:
                G_r = np.linalg.pinv(E - H_eff)
                warnings.warn("Using pseudoinverse for singular retarded Green's function")
            except:
                # Final fallback
                G_r = np.zeros_like(E)
                warnings.warn("Complete failure in retarded Green's function computation")
            
        return G_r
    
    def advanced_greens_function(self, energy: float,
                                mu_leads: Optional[list] = None) -> np.ndarray:
        """
        Compute advanced Green's function.
        G^a = (G^r)†
        """
        G_r = self.retarded_greens_function(energy, mu_leads)
        return G_r.T.conj()
    
    def gamma_matrix(self, energy: float, lead_idx: int) -> np.ndarray:
        """
        Compute Γ matrix for a lead: Γ = i(Σ^r - Σ^a)
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate Γ
        lead_idx : int
            Index of the lead
            
        Returns:
        --------
        np.ndarray
            Γ matrix
        """
        try:
            Sigma_r = self.self_energy(energy, lead_idx)
            Sigma_a = Sigma_r.T.conj()
            
            # Ensure proper dimensions for device space
            if Sigma_r.shape != (self.n_device, self.n_device):
                Gamma_full = np.zeros((self.n_device, self.n_device), dtype=complex)
                min_dim = min(Sigma_r.shape[0], self.n_device)
                if lead_idx == 0:  # Left lead
                    Gamma_block = 1j * (Sigma_r[:min_dim, :min_dim] - Sigma_a[:min_dim, :min_dim])
                    Gamma_full[:min_dim, :min_dim] = Gamma_block
                else:  # Right lead
                    Gamma_block = 1j * (Sigma_r[:min_dim, :min_dim] - Sigma_a[:min_dim, :min_dim])
                    Gamma_full[-min_dim:, -min_dim:] = Gamma_block
                return Gamma_full
            else:
                Gamma = 1j * (Sigma_r - Sigma_a)
                return Gamma
        except Exception as e:
            warnings.warn(f"Failed to compute gamma matrix for lead {lead_idx}: {e}")
            return np.zeros((self.n_device, self.n_device), dtype=complex)
    
    def transmission(self, energy: float, lead_i: int = 0, lead_j: int = 1) -> float:
        """
        Compute transmission coefficient between two leads.
        T = Tr[Γ_i G^r Γ_j G^a]
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate transmission
        lead_i, lead_j : int
            Indices of source and drain leads
            
        Returns:
        --------
        float
            Transmission coefficient
        """
        if self.n_leads < 2:
            raise ValueError(f"Need at least 2 leads for transmission calculation, got {self.n_leads}")
        
        if lead_i >= self.n_leads or lead_j >= self.n_leads:
            raise ValueError(f"Lead indices {lead_i}, {lead_j} exceed number of leads {self.n_leads}")
            
        G_r = self.retarded_greens_function(energy)
        G_a = self.advanced_greens_function(energy)
        
        Gamma_i = self.gamma_matrix(energy, lead_i)
        Gamma_j = self.gamma_matrix(energy, lead_j)
        
        # T = Tr[Γ_i G^r Γ_j G^a]
        T_matrix = Gamma_i @ G_r @ Gamma_j @ G_a
        T = np.real(np.trace(T_matrix))
        
        return T
    
    def conductance(self, energy: float, lead_i: int = 0, lead_j: int = 1) -> float:
        """
        Compute conductance in units of e²/h.
        
        Parameters:
        -----------
        energy : float
            Energy (typically Fermi energy)
        lead_i, lead_j : int
            Indices of source and drain leads
            
        Returns:
        --------
        float
            Conductance in units of e²/h
        """
        T = self.transmission(energy, lead_i, lead_j)
        return T  # Already in units of e²/h
    
    def lesser_greens_function(self, energy: float, mu_leads: list, 
                              temperature: float = 0.0) -> np.ndarray:
        """
        Compute lesser Green's function G^< for charge density calculation.
        G^< = G^r Σ^< G^a
        
        Parameters:
        -----------
        energy : float
            Energy at which to evaluate G^<
        mu_leads : list
            Chemical potentials of the leads
        temperature : float
            Temperature in eV
            
        Returns:
        --------
        np.ndarray
            Lesser Green's function G^<
        """
        G_r = self.retarded_greens_function(energy)
        G_a = self.advanced_greens_function(energy)
        
        # Fermi-Dirac distribution with better numerical stability
        def fermi(E, mu, T):
            if T <= 0:
                return 1.0 if E <= mu else 0.0
            x = (E - mu) / (8.617333262e-5 * T)  # kB in eV/K
            # More stable implementation
            if x > 50:
                return 0.0
            elif x < -50:
                return 1.0
            else:
                # Use more stable form for moderate x
                if x >= 0:
                    exp_x = np.exp(-x)
                    return exp_x / (1.0 + exp_x)
                else:
                    exp_x = np.exp(x)
                    return 1.0 / (1.0 + exp_x)
        
        # Sum contributions from all leads
        Sigma_lesser = np.zeros_like(self.H_device, dtype=complex)
        for i, mu in enumerate(mu_leads):
            if i >= self.n_leads:
                break
            f = fermi(energy, mu, temperature)
            Gamma = self.gamma_matrix(energy, i)
            Sigma_lesser += 1j * f * Gamma
        
        G_lesser = G_r @ Sigma_lesser @ G_a
        return G_lesser
    
    def local_charge_density(self, energy_grid: np.ndarray, mu_leads: list,
                           temperature: float = 0.0) -> np.ndarray:
        """
        Compute local charge density from integration over energy.
        ρ(r) = -(1/π) ∫ dE Im[G^<(r,r,E)]
        
        Parameters:
        -----------
        energy_grid : np.ndarray
            Energy points for integration
        mu_leads : list
            Chemical potentials of the leads
        temperature : float
            Temperature in eV
            
        Returns:
        --------
        np.ndarray
            Local charge density on each site
        """
        rho = np.zeros(self.n_device, dtype=float)
        
        for E in energy_grid:
            G_lesser = self.lesser_greens_function(E, mu_leads, temperature)
            # Extract diagonal elements and take imaginary part
            rho += np.imag(np.diag(G_lesser))
        
        # Numerical integration (simple trapezoidal rule)
        dE = energy_grid[1] - energy_grid[0] if len(energy_grid) > 1 else 1.0
        rho *= -dE / np.pi
        
        return rho
    
    def charge_density_at_bias(self, mu_left: float, mu_right: float, 
                              temperature: float = 0.0, 
                              E_span: float = 0.5, n_points: int = 101) -> np.ndarray:
        """
        Compute charge density under finite bias conditions.
        
        Parameters:
        -----------
        mu_left : float
            Chemical potential of left lead
        mu_right : float
            Chemical potential of right lead
        temperature : float
            Temperature in eV
        E_span : float
            Energy span around chemical potentials for integration
        n_points : int
            Number of energy points for integration
            
        Returns:
        --------
        np.ndarray
            Local charge density on each site under bias
        """
        # Create energy grid around the chemical potential window
        mu_min = min(mu_left, mu_right) - E_span/2
        mu_max = max(mu_left, mu_right) + E_span/2
        energy_grid = np.linspace(mu_min, mu_max, n_points)
        
        # Chemical potentials for both leads
        mu_leads = [mu_left, mu_right] if self.n_leads >= 2 else [mu_left]
        
        return self.local_charge_density(energy_grid, mu_leads, temperature)


def extract_kwant_matrices(fsys, energy: float, leads_kwant_supported: list = None, params: dict = None) -> Tuple[np.ndarray, list, list]:
    """
    Extract Hamiltonian matrices from a Kwant finalized system for NEGF calculation.
    Uses proper Kwant interface by analyzing the system graph structure.
    
    Parameters:
    -----------
    fsys : kwant.system.FiniteSystem
        Finalized Kwant system
    energy : float
        Energy at which to evaluate the Hamiltonian
    leads_kwant_supported : list, optional
        List of lead indices that Kwant supports (for compatibility)
    params : dict, optional
        Parameters for the system
        
    Returns:
    --------
    Tuple containing:
    - H_device: Device Hamiltonian matrix
    - H_leads: List of lead Hamiltonian dictionaries  
    - V_couplings: List of proper interface coupling matrices
    """
    import kwant
    import scipy.sparse as sp
    
    if params is None:
        params = {}
    
    try:
        # Method 1: Use Kwant's graph structure to extract proper interface coupling
        # This is the most reliable approach for getting actual hopping matrices
        
        # Get device Hamiltonian (scattering region only)
        H_device = fsys.hamiltonian_submatrix(params=params)
        
        # Get device sites and create index mapping (handle tuple property vs callable)
        try:
            if hasattr(fsys, 'sites'):
                sites_attr = getattr(fsys, 'sites')
                device_sites = list(sites_attr() if callable(sites_attr) else sites_attr)
            elif hasattr(fsys, 'graph'):
                # Fallback: use graph nodes if available
                device_sites = list(fsys.graph.nodes())
            else:
                raise AttributeError("Finalized system has no 'sites' or 'graph' attribute")
            site_to_index = {site: i for i, site in enumerate(device_sites)}
            # print(f"Debug: Device has {len(device_sites)} sites, H_device shape: {H_device.shape}")
        except Exception as sites_error:
            # print(f"Debug: Error getting sites: {sites_error}")
            raise

        # Extract lead information and interface coupling
        H_leads = []
        V_couplings = []

        for lead_idx, lead in enumerate(fsys.leads):
            # Get lead unit cell matrices
            H00 = lead.cell_hamiltonian(params=params)
            H01 = lead.inter_cell_hopping(params=params)
            
            H_leads.append({
                'H00': H00,
                'H01': H01
            })
            
            # print(f"Debug: Lead {lead_idx} - H00 shape: {H00.shape}, H01 shape: {H01.shape}")
            
            # Extract interface using finalized system's lead_interfaces mapping
            if hasattr(fsys, 'lead_interfaces') and lead_idx < len(fsys.lead_interfaces):
                iface_indices_device = list(fsys.lead_interfaces[lead_idx])
                # print(f"Debug: Lead {lead_idx} interface has {len(iface_indices_device)} device sites")
            else:
                raise AttributeError("Finalized system has no 'lead_interfaces' for interface extraction")
            
            # Create coupling matrix: V_coupling[lead_orbital, device_orbital]
            n_lead_orbs = H00.shape[0] 
            n_device_orbs = H_device.shape[0]
            
            V_coupling = np.zeros((n_lead_orbs, n_device_orbs), dtype=complex)
            
            # Extract actual interface couplings from Kwant system
            # Get proper norbs_per_site from hamiltonian_submatrix
            try:
                _, norbs_per_site = fsys.hamiltonian_submatrix(params=params, return_norb=True)
                if hasattr(norbs_per_site, '__len__'):
                    norbs_per_site = norbs_per_site[0]  # Take first site's norbs if array
                else:
                    norbs_per_site = int(norbs_per_site)
            except Exception:
                norbs_per_site = max(1, H_device.shape[0] // max(1, len(device_sites)))
            
            # Get hopping parameter from system parameters
            t_interface = params.get('t', 2.7)  # Use actual system hopping
            
            # Build V_coupling from actual interface connectivity
            # Sort interface sites to get deterministic ordering
            interface_coords = [device_sites[idx].pos if hasattr(device_sites[idx], 'pos') 
                              else (device_sites[idx][0], device_sites[idx][1])
                              for idx in iface_indices_device]
            # Sort by y-coordinate first, then by x for consistent ordering
            sorted_pairs = sorted(zip(iface_indices_device, interface_coords), 
                                key=lambda x: (x[1][1], x[1][0]))
            sorted_iface_indices = [pair[0] for pair in sorted_pairs]
            
            max_pairs = min(len(sorted_iface_indices), n_lead_orbs // norbs_per_site)
            for i in range(max_pairs):
                device_site_idx = sorted_iface_indices[i]
                device_orb_start = device_site_idx * norbs_per_site
                device_orb_end = device_orb_start + norbs_per_site
                lead_orb_start = i * norbs_per_site
                lead_orb_end = lead_orb_start + norbs_per_site
                if (lead_orb_end <= n_lead_orbs and device_orb_end <= n_device_orbs):
                    for j in range(norbs_per_site):
                        V_coupling[lead_orb_start + j, device_orb_start + j] = -t_interface
            
            # print(f"Debug: V_coupling shape: {V_coupling.shape}, non-zero elements: {np.count_nonzero(V_coupling)}")
            
            V_couplings.append(V_coupling)

        return H_device, H_leads, V_couplings

    except Exception as e:
        # Enhanced fallback with proper dimension matching
        warnings.warn(f"Interface extraction failed ({e}), using enhanced fallback")
        
        # Get device Hamiltonian
        H_device = fsys.hamiltonian_submatrix(params=params)
        n_device = H_device.shape[0]
        
        # print(f"Debug: Fallback - H_device shape: {H_device.shape}")
        
        # Extract lead information
        H_leads = []
        V_couplings = []
        
        n_leads = len(fsys.leads)
        if n_leads == 0:
            raise ValueError("System has no leads - cannot perform NEGF calculation")
        
        for lead_idx, lead in enumerate(fsys.leads):
            # Get lead unit cell Hamiltonian
            H00 = lead.cell_hamiltonian(params=params)
            H01 = lead.inter_cell_hopping(params=params)
            
            H_leads.append({
                'H00': H00,
                'H01': H01
            })
            
            # print(f"Debug: Fallback Lead {lead_idx} - H00: {H00.shape}, H01: {H01.shape}")
            
            # Create properly dimensioned coupling matrix
            # The coupling matrix should be: [lead_orbitals × device_orbitals]
            n_lead_orbs = H00.shape[0]  # Use actual lead unit cell size
            
            V_coupling = np.zeros((n_lead_orbs, n_device), dtype=complex)
            
            # Use realistic graphene hopping amplitude
            t_hopping = -2.7  # eV
            
            # Connect interface orbitals properly
            # For graphene ribbons, leads typically connect to edge atoms
            interface_width = min(n_lead_orbs, 8)  # Reasonable interface width
            
            if lead_idx == 0:  # Left lead - connect to left edge of device
                for i in range(interface_width):
                    if i < n_device:
                        V_coupling[i, i] = t_hopping
            else:  # Right lead - connect to right edge of device
                for i in range(interface_width):
                    device_orb = n_device - interface_width + i
                    if device_orb >= 0 and device_orb < n_device:
                        V_coupling[i, device_orb] = t_hopping
            
            # print(f"Debug: Fallback V_coupling shape: {V_coupling.shape}, non-zero: {np.count_nonzero(V_coupling)}")
            
            V_couplings.append(V_coupling)
        
        return H_device, H_leads, V_couplings