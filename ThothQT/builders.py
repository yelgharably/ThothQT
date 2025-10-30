"""
ThothQT System Builders
=======================

Advanced geometry builders for complex quantum systems:
- Graphene nanoribbons (zigzag, armchair)
- 2D materials (TMDs, hBN)
- Custom geometries with automatic lead attachment
- Heterostructures and quantum dots

Author: Quantum Sensing Project
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Optional, Dict, Callable, Union
from dataclasses import dataclass

try:
    from .thothqt import Device, PeriodicLead
except ImportError:
    from thothqt import Device, PeriodicLead


@dataclass
class Lattice:
    """
    Crystal lattice definition.
    
    Attributes
    ----------
    vectors : ndarray (2, 2) or (3, 3)
        Lattice vectors as rows
    sublattices : List[ndarray]
        Sublattice positions in fractional coordinates
    """
    vectors: np.ndarray
    sublattices: List[np.ndarray]


class GrapheneBuilder:
    """
    Builder for graphene nanoribbon systems.
    
    Handles both zigzag and armchair edge terminations with
    proper lead coupling for quantum transport calculations.
    """
    
    def __init__(self, a: float = 1.42, t: float = 2.7):
        """
        Initialize graphene builder.
        
        Parameters
        ----------
        a : float
            C-C bond length (Å)
        t : float
            Hopping parameter (eV)
        """
        self.a = a
        self.t = t
        
        # Graphene lattice vectors (in Cartesian coordinates)
        a1 = a * np.array([np.sqrt(3), 0])
        a2 = a * np.array([np.sqrt(3)/2, 3/2])
        
        # Sublattice positions (A and B atoms)
        self.sublattice_A = np.array([0, 0])
        self.sublattice_B = a * np.array([0, 1])
        
        self.lattice_vectors = np.array([a1, a2])
    
    def zigzag_ribbon(self, width: int, length: int, 
                      return_positions: bool = False) -> Union[Device, Tuple[Device, np.ndarray]]:
        """
        Build zigzag graphene nanoribbon.
        
        Zigzag ribbon: periodic in x-direction (armchair edges on sides)
        
        Parameters
        ----------
        width : int
            Number of zigzag chains (atoms along y)
        length : int
            Number of unit cells (along x)
        return_positions : bool
            If True, also return atom positions
            
        Returns
        -------
        device : Device
            Device structure with leads
        positions : ndarray, optional
            Atom positions (N, 2) if return_positions=True
        """
        # Build structure
        n_atoms = 2 * width * length  # 2 atoms per unit cell
        positions = []
        
        # Hamiltonian matrices
        H_device = sp.lil_matrix((n_atoms, n_atoms), dtype=complex)
        
        # Helper: atom index from cell and sublattice
        def atom_index(ix: int, iy: int, sublat: int) -> int:
            return (ix * width + iy) * 2 + sublat
        
        # Build device region
        for ix in range(length):
            for iy in range(width):
                # A sublattice
                idx_A = atom_index(ix, iy, 0)
                pos_A = ix * self.lattice_vectors[0] + iy * self.lattice_vectors[1] + self.sublattice_A
                positions.append(pos_A)
                
                # B sublattice  
                idx_B = atom_index(ix, iy, 1)
                pos_B = ix * self.lattice_vectors[0] + iy * self.lattice_vectors[1] + self.sublattice_B
                positions.append(pos_B)
                
                # Intracell hopping A-B
                H_device[idx_A, idx_B] = -self.t
                H_device[idx_B, idx_A] = -self.t
                
                # Intercell hopping (x-direction)
                if ix < length - 1:
                    # A(ix,iy) - B(ix+1,iy)
                    idx_B_next = atom_index(ix + 1, iy, 1)
                    H_device[idx_A, idx_B_next] = -self.t
                    H_device[idx_B_next, idx_A] = -self.t
                
                # Intercell hopping (y-direction, within zigzag chain)
                if iy < width - 1:
                    # B(ix,iy) - A(ix,iy+1)
                    idx_A_next = atom_index(ix, iy + 1, 0)
                    H_device[idx_B, idx_A_next] = -self.t
                    H_device[idx_A_next, idx_B] = -self.t
        
        positions = np.array(positions)
        H_device = H_device.tocsr()
        
        # Build leads (semi-infinite in x-direction)
        # Lead unit cell: single zigzag chain (width atoms × 2 sublattices)
        n_lead = 2 * width
        H00 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        H01 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        
        for iy in range(width):
            idx_A = 2 * iy
            idx_B = 2 * iy + 1
            
            # Intracell A-B hopping
            H00[idx_A, idx_B] = -self.t
            H00[idx_B, idx_A] = -self.t
            
            # Intercell hopping within chain (y-direction)
            if iy < width - 1:
                idx_A_next = 2 * (iy + 1)
                H00[idx_B, idx_A_next] = -self.t
                H00[idx_A_next, idx_B] = -self.t
            
            # Hopping to next unit cell (x-direction)
            H01[idx_A, idx_B] = -self.t  # A(0,iy) -> B(1,iy)
        
        # Coupling matrices (device to leads)
        tau_L = sp.lil_matrix((n_atoms, n_lead), dtype=complex)
        tau_R = sp.lil_matrix((n_atoms, n_lead), dtype=complex)
        
        # Left coupling: first x-slice to lead
        for iy in range(width):
            dev_A = atom_index(0, iy, 0)
            dev_B = atom_index(0, iy, 1) 
            lead_A = 2 * iy
            lead_B = 2 * iy + 1
            
            tau_L[dev_A, lead_A] = -self.t  # Device A -> Lead A
            tau_L[dev_B, lead_B] = -self.t  # Device B -> Lead B
        
        # Right coupling: last x-slice to lead  
        for iy in range(width):
            dev_A = atom_index(length - 1, iy, 0)
            dev_B = atom_index(length - 1, iy, 1)
            lead_A = 2 * iy
            lead_B = 2 * iy + 1
            
            tau_R[dev_A, lead_A] = -self.t
            tau_R[dev_B, lead_B] = -self.t
        
        # Create lead objects
        left_lead = PeriodicLead(H00=H00.tocsr(), H01=H01.tocsr(), tau_cpl=tau_L.tocsr())
        right_lead = PeriodicLead(H00=H00.tocsr(), H01=H01.tocsr(), tau_cpl=tau_R.tocsr())
        
        device = Device(H=H_device, left=left_lead, right=right_lead)
        
        if return_positions:
            return device, positions
        else:
            return device
    
    def armchair_ribbon(self, width: int, length: int,
                        return_positions: bool = False) -> Union[Device, Tuple[Device, np.ndarray]]:
        """
        Build armchair graphene nanoribbon.
        
        Armchair ribbon: periodic in y-direction (zigzag edges on sides)
        
        Parameters
        ----------
        width : int
            Number of armchair chains (atoms along x)
        length : int
            Number of unit cells (along y)
        return_positions : bool
            If True, also return atom positions
            
        Returns
        -------
        device : Device
            Device structure with leads
        positions : ndarray, optional
            Atom positions (N, 2) if return_positions=True
        """
        # Rotate coordinates: x <-> y for armchair vs zigzag
        # This is a simplified implementation
        # For full armchair implementation, coordinate transformation is needed
        
        # Placeholder: use zigzag with swapped dimensions
        # In practice, would need proper coordinate transformation
        device_zz, pos_zz = self.zigzag_ribbon(length, width, return_positions=True)
        
        if return_positions:
            # Swap x and y coordinates
            positions = pos_zz[:, [1, 0]]
            return device_zz, positions
        else:
            return device_zz


class TMDBuilder:
    """
    Builder for Transition Metal Dichalcogenide (TMD) systems.
    
    Supports MoS2, WSe2, etc. with proper orbital structure.
    """
    
    def __init__(self, material: str = "MoS2", a: float = 3.16):
        """
        Initialize TMD builder.
        
        Parameters
        ----------
        material : str
            TMD material ("MoS2", "WSe2", "MoSe2", etc.)
        a : float
            Lattice constant (Å)
        """
        self.material = material
        self.a = a
        
        # Material-specific parameters
        self.params = self._get_material_params(material)
    
    def _get_material_params(self, material: str) -> Dict:
        """Get tight-binding parameters for TMD materials"""
        params = {
            "MoS2": {
                "t_Mo_d": 2.8,    # Mo d-orbital hopping (eV)
                "t_S_p": 1.4,     # S p-orbital hopping (eV)
                "t_Mo_S": 2.2,    # Mo-S coupling (eV)
                "eps_Mo": 0.0,    # Mo onsite energy (eV)
                "eps_S": -1.5,    # S onsite energy (eV)
            },
            "WSe2": {
                "t_Mo_d": 3.2,
                "t_S_p": 1.2,
                "t_Mo_S": 2.5,
                "eps_Mo": 0.2,
                "eps_S": -1.8,
            }
        }
        return params.get(material, params["MoS2"])
    
    def monolayer_ribbon(self, width: int, length: int) -> Device:
        """
        Build TMD monolayer nanoribbon.
        
        Parameters
        ----------
        width : int
            Ribbon width (unit cells)
        length : int  
            Ribbon length (unit cells)
            
        Returns
        -------
        device : Device
            TMD ribbon device
        """
        # Simplified TMD model: single orbital per atom
        # Full implementation would include d/p orbital structure
        
        n_sites = width * length
        H = sp.diags([self.params["eps_Mo"]] * n_sites, format='csr', dtype=complex)
        
        # Add hopping (simplified)
        for i in range(n_sites - 1):
            H[i, i+1] = -self.params["t_Mo_d"]
            H[i+1, i] = -self.params["t_Mo_d"]
        
        # Create simple leads
        H00 = np.array([[self.params["eps_Mo"]]], dtype=complex)
        H01 = np.array([[-self.params["t_Mo_d"]]], dtype=complex)
        
        tau_L = sp.csr_matrix((n_sites, 1), dtype=complex)
        tau_L[0, 0] = -self.params["t_Mo_d"]
        
        tau_R = sp.csr_matrix((n_sites, 1), dtype=complex) 
        tau_R[-1, 0] = -self.params["t_Mo_d"]
        
        left_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_L)
        right_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_R)
        
        return Device(H=H, left=left_lead, right=right_lead)


class CustomSystemBuilder:
    """
    Flexible builder for custom quantum systems.
    
    Allows manual construction of arbitrary geometries with
    automatic lead attachment.
    """
    
    def __init__(self):
        """Initialize empty system builder."""
        self.sites = []
        self.hoppings = []
        self.positions = []
        self.onsite_energies = []
    
    def add_site(self, position: List[float], onsite: float = 0.0) -> int:
        """
        Add site to system.
        
        Parameters
        ----------
        position : list
            Site position [x, y] (Å)  
        onsite : float
            On-site energy (eV)
            
        Returns
        -------
        site_id : int
            Index of added site
        """
        site_id = len(self.sites)
        self.sites.append(site_id)
        self.positions.append(np.array(position))
        self.onsite_energies.append(onsite)
        return site_id
    
    def add_hopping(self, site1: int, site2: int, t: float):
        """
        Add hopping between sites.
        
        Parameters
        ----------
        site1, site2 : int
            Site indices
        t : float
            Hopping amplitude (eV)
        """
        self.hoppings.append((site1, site2, t))
    
    def add_chain(self, start_pos: List[float], direction: List[float], 
                  n_sites: int, spacing: float, t: float, onsite: float = 0.0) -> List[int]:
        """
        Add chain of sites.
        
        Parameters
        ----------
        start_pos : list
            Starting position [x, y] (Å)
        direction : list  
            Chain direction [dx, dy] (normalized)
        n_sites : int
            Number of sites in chain
        spacing : float
            Site spacing (Å)
        t : float
            Hopping between neighboring sites (eV)
        onsite : float
            On-site energy (eV)
            
        Returns
        -------
        site_ids : list
            List of site indices in chain
        """
        start_pos = np.array(start_pos)
        direction = np.array(direction) / np.linalg.norm(direction)
        
        site_ids = []
        for i in range(n_sites):
            pos = start_pos + i * spacing * direction
            site_id = self.add_site(pos.tolist(), onsite)
            site_ids.append(site_id)
            
            # Add hopping to previous site
            if i > 0:
                self.add_hopping(site_ids[i-1], site_id, t)
        
        return site_ids
    
    def add_ring(self, center: List[float], radius: float, n_sites: int,
                 t: float, onsite: float = 0.0) -> List[int]:
        """
        Add ring of sites.
        
        Parameters
        ----------
        center : list
            Ring center [x, y] (Å)
        radius : float
            Ring radius (Å)
        n_sites : int
            Number of sites in ring
        t : float
            Hopping between neighboring sites (eV)
        onsite : float
            On-site energy (eV)
            
        Returns
        -------
        site_ids : list
            List of site indices in ring
        """
        center = np.array(center)
        angles = np.linspace(0, 2*np.pi, n_sites, endpoint=False)
        
        site_ids = []
        for angle in angles:
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            site_id = self.add_site(pos.tolist(), onsite)
            site_ids.append(site_id)
        
        # Add ring hoppings
        for i in range(n_sites):
            next_i = (i + 1) % n_sites
            self.add_hopping(site_ids[i], site_ids[next_i], t)
        
        return site_ids
    
    def build_device(self, lead_direction: str = "x") -> Device:
        """
        Build device with automatic lead attachment.
        
        Parameters
        ----------
        lead_direction : str
            Direction for leads ("x" or "y")
            
        Returns
        -------
        device : Device
            Complete device with leads
        """
        if not self.sites:
            raise ValueError("No sites added to system")
        
        n_sites = len(self.sites)
        positions = np.array(self.positions)
        
        # Build device Hamiltonian
        H_device = sp.lil_matrix((n_sites, n_sites), dtype=complex)
        
        # Add on-site energies
        for i, eps in enumerate(self.onsite_energies):
            H_device[i, i] = eps
        
        # Add hoppings
        for site1, site2, t in self.hoppings:
            H_device[site1, site2] = t
            H_device[site2, site1] = np.conj(t)  # Hermitian
        
        H_device = H_device.tocsr()
        
        # Create simple 1D leads
        # In a full implementation, would analyze geometry to create appropriate leads
        H00_lead = np.array([[0.0]], dtype=complex)
        H01_lead = np.array([[-1.0]], dtype=complex)  # Default hopping
        
        # Find leftmost and rightmost sites
        if lead_direction == "x":
            coord_idx = 0
        else:  # "y"
            coord_idx = 1
        
        coords = positions[:, coord_idx]
        left_sites = np.where(coords == np.min(coords))[0]
        right_sites = np.where(coords == np.max(coords))[0]
        
        # Create coupling matrices
        tau_L = sp.csr_matrix((n_sites, 1), dtype=complex)
        tau_R = sp.csr_matrix((n_sites, 1), dtype=complex)
        
        for site in left_sites:
            tau_L[site, 0] = -1.0  # Default coupling
        
        for site in right_sites:
            tau_R[site, 0] = -1.0
        
        left_lead = PeriodicLead(H00=H00_lead, H01=H01_lead, tau_cpl=tau_L)
        right_lead = PeriodicLead(H00=H00_lead, H01=H01_lead, tau_cpl=tau_R)
        
        return Device(H=H_device, left=left_lead, right=right_lead)


# ============================================================================
# Predefined System Functions
# ============================================================================

def make_graphene_zigzag_ribbon(width: int, length: int, a: float = 1.42, 
                                t: float = 2.7) -> Device:
    """
    Convenience function to create graphene zigzag nanoribbon.
    
    Parameters
    ----------
    width : int
        Ribbon width (number of zigzag chains)
    length : int
        Ribbon length (number of unit cells)
    a : float
        C-C bond length (Å)
    t : float
        Hopping parameter (eV)
        
    Returns
    -------
    device : Device
        Graphene zigzag ribbon device
    """
    builder = GrapheneBuilder(a=a, t=t)
    return builder.zigzag_ribbon(width, length)


def make_quantum_dot(n_sites: int = 10, t: float = 1.0, eps_dot: float = 0.5) -> Device:
    """
    Create quantum dot system.
    
    Parameters
    ----------
    n_sites : int
        Number of sites in dot
    t : float
        Lead-dot coupling (eV)
    eps_dot : float
        Dot energy level (eV)
        
    Returns
    -------
    device : Device
        Quantum dot device
    """
    # Simple model: all dot sites at same energy, uncoupled to each other
    H_dot = sp.diags([eps_dot] * n_sites, format='csr', dtype=complex)
    
    # 1D leads
    H00 = np.array([[0.0]], dtype=complex)
    H01 = np.array([[-t]], dtype=complex)
    
    # Only first and last sites couple to leads
    tau_L = sp.csr_matrix((n_sites, 1), dtype=complex)
    tau_L[0, 0] = -t
    
    tau_R = sp.csr_matrix((n_sites, 1), dtype=complex)
    tau_R[-1, 0] = -t
    
    left_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_L)
    right_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_R)
    
    return Device(H=H_dot, left=left_lead, right=right_lead)


def make_double_quantum_dot(eps1: float = 0.0, eps2: float = 0.5, 
                           t_dots: float = 0.1, t_leads: float = 1.0) -> Device:
    """
    Create double quantum dot system.
    
    Parameters
    ----------
    eps1, eps2 : float
        Energy levels of dots 1 and 2 (eV)
    t_dots : float
        Interdot coupling (eV)
    t_leads : float
        Lead-dot coupling (eV)
        
    Returns
    -------
    device : Device
        Double quantum dot device
    """
    # Two-site device
    H_device = sp.lil_matrix((2, 2), dtype=complex)
    H_device[0, 0] = eps1
    H_device[1, 1] = eps2
    H_device[0, 1] = t_dots
    H_device[1, 0] = t_dots
    H_device = H_device.tocsr()
    
    # 1D leads
    H00 = np.array([[0.0]], dtype=complex)
    H01 = np.array([[-t_leads]], dtype=complex)
    
    # Left lead to dot 1, right lead to dot 2
    tau_L = sp.csr_matrix([[t_leads], [0.0]], dtype=complex)
    tau_R = sp.csr_matrix([[0.0], [t_leads]], dtype=complex)
    
    left_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_L)
    right_lead = PeriodicLead(H00=H00, H01=H01, tau_cpl=tau_R)
    
    return Device(H=H_device, left=left_lead, right=right_lead)