"""
ThoothQT System Builders
========================

Advanced geometry builders for complex quantum systems:
- Graphene nanoribbons (zigzag, armchair)
- 2D materials (TMDs, hBN)
- Heterostructures
- Custom geometries with automatic lead attachment

Author: Quantum Sensing Project
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass

from toothqt_production import PeriodicLead, Device


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
    proper lead coupling.
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
        
        # Graphene lattice vectors
        a1 = a * np.array([np.sqrt(3), 0])
        a2 = a * np.array([np.sqrt(3)/2, 3/2])
        
        # Sublattice positions (in Cartesian coordinates)
        self.sublattice_A = np.array([0, 0])
        self.sublattice_B = a * np.array([0, 1])
        
        self.lattice_vectors = np.array([a1, a2])
    
    def zigzag_ribbon(self, width: int, length: int, 
                      return_positions: bool = False) -> Tuple[Device, Optional[np.ndarray]]:
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
        positions = np.zeros((n_atoms, 2))
        
        # Hamiltonian
        H = sp.lil_matrix((n_atoms, n_atoms), dtype=complex)
        
        # Helper: atom index from cell and position
        def idx(cell_x, chain_y, sublattice):
            """Get global atom index."""
            return cell_x * (2 * width) + chain_y * 2 + sublattice
        
        # Build structure unit cell by unit cell
        for ix in range(length):
            for iy in range(width):
                # Position in cell
                x0 = ix * self.a * np.sqrt(3)
                y0 = iy * self.a * 3
                
                i_A = idx(ix, iy, 0)
                i_B = idx(ix, iy, 1)
                
                # Positions
                positions[i_A] = [x0, y0]
                positions[i_B] = [x0, y0 + self.a]
                
                # Intra-cell hopping (A-B vertical)
                H[i_A, i_B] = -self.t
                H[i_B, i_A] = -self.t
                
                # Hopping to next chain (diagonal bonds)
                if iy < width - 1:
                    i_A_next = idx(ix, iy + 1, 0)
                    # B to A_next (lower right)
                    H[i_B, i_A_next] = -self.t
                    H[i_A_next, i_B] = -self.t
                    
                    # B to next A_next via second neighbor
                    i_B_next = idx(ix, iy + 1, 1)
                    # A to B_next (upper right)  
                    H[i_A, i_B_next] = -self.t
                    H[i_B_next, i_A] = -self.t
                
                # Hopping to next cell (along x)
                # For zigzag: only horizontal bonds between cells
                if ix < length - 1:
                    j_A = idx(ix + 1, iy, 0)
                    j_B = idx(ix + 1, iy, 1)
                    
                    # A to next A (horizontal)
                    H[i_A, j_A] = -self.t
                    H[j_A, i_A] = -self.t
                    
                    # B to next B (horizontal)
                    H[i_B, j_B] = -self.t
                    H[j_B, i_B] = -self.t
        
        H = H.tocsr()
        
        # Build lead (single unit cell)
        n_lead = 2 * width
        H00_lead = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        H01_lead = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        
        for iy in range(width):
            i_A = iy * 2
            i_B = iy * 2 + 1
            
            # Intra-cell hoppings (same as device)
            H00_lead[i_A, i_B] = -self.t
            H00_lead[i_B, i_A] = -self.t
            
            if iy < width - 1:
                i_A_next = (iy + 1) * 2
                i_B_next = (iy + 1) * 2 + 1
                H00_lead[i_B, i_A_next] = -self.t
                H00_lead[i_A_next, i_B] = -self.t
                H00_lead[i_A, i_B_next] = -self.t
                H00_lead[i_B_next, i_A] = -self.t
            
            # Inter-cell hoppings (between unit cells along x)
            # For zigzag ribbon: only horizontal bonds (no diagonal inter-cell)
            # Each atom connects to its equivalent in the next cell
            H01_lead[i_A, i_A] = -self.t  # A → A horizontal
            H01_lead[i_B, i_B] = -self.t  # B → B horizontal
        
        H00_lead = H00_lead.toarray()
        H01_lead = H01_lead.toarray()
        
        # Create leads
        lead_L = PeriodicLead(H00=H00_lead, H01=H01_lead)
        lead_R = PeriodicLead(H00=H00_lead, H01=H01_lead)
        
        # Create device
        device = Device(H=H, S=None, left=lead_L, right=lead_R, Ef=0.0)
        
        if return_positions:
            return device, positions
        return device, None
    
    def armchair_ribbon(self, width: int, length: int,
                       return_positions: bool = False) -> Tuple[Device, Optional[np.ndarray]]:
        """
        Build armchair graphene nanoribbon.
        
        Armchair ribbon: periodic in y-direction (zigzag edges on top/bottom)
        
        Parameters
        ----------
        width : int
            Number of dimers across width
        length : int
            Number of unit cells along transport
        return_positions : bool
            If True, return atom positions
            
        Returns
        -------
        device : Device
            Device with leads
        positions : ndarray, optional
            Atom positions if requested
        """
        # Armchair: 2 atoms per dimer, width dimers = 2*width atoms per unit cell
        n_per_cell = 2 * width
        n_atoms = n_per_cell * length
        
        H = sp.lil_matrix((n_atoms, n_atoms), dtype=complex)
        positions = np.zeros((n_atoms, 2))
        
        def idx(cell, dimer, sublattice):
            return cell * n_per_cell + dimer * 2 + sublattice
        
        # Build device
        for cell in range(length):
            for dimer in range(width):
                i1 = idx(cell, dimer, 0)
                i2 = idx(cell, dimer, 1)
                
                x0 = cell * 3 * self.a
                y0 = dimer * np.sqrt(3) * self.a
                
                positions[i1] = [x0, y0]
                positions[i2] = [x0 + self.a, y0]
                
                # Intra-dimer hopping
                H[i1, i2] = -self.t
                H[i2, i1] = -self.t
                
                # Vertical connections
                if dimer < width - 1:
                    i3 = idx(cell, dimer + 1, 0)
                    H[i1, i3] = -self.t
                    H[i3, i1] = -self.t
                    H[i2, i3] = -self.t
                    H[i3, i2] = -self.t
                
                # Horizontal connections
                if cell < length - 1:
                    j1 = idx(cell + 1, dimer, 0)
                    j2 = idx(cell + 1, dimer, 1)
                    H[i1, j1] = -self.t
                    H[j1, i1] = -self.t
                    H[i2, j2] = -self.t
                    H[j2, i2] = -self.t
        
        H = H.tocsr()
        
        # Build leads
        H00 = np.zeros((n_per_cell, n_per_cell), dtype=complex)
        H01 = np.zeros((n_per_cell, n_per_cell), dtype=complex)
        
        for dimer in range(width):
            i1 = dimer * 2
            i2 = dimer * 2 + 1
            
            H00[i1, i2] = -self.t
            H00[i2, i1] = -self.t
            
            if dimer < width - 1:
                i3 = (dimer + 1) * 2
                H00[i1, i3] = -self.t
                H00[i3, i1] = -self.t
                H00[i2, i3] = -self.t
                H00[i3, i2] = -self.t
            
            H01[i1, i1] = -self.t
            H01[i2, i2] = -self.t
        
        lead_L = PeriodicLead(H00=H00, H01=H01)
        lead_R = PeriodicLead(H00=H00, H01=H01)
        
        device = Device(H=H, S=None, left=lead_L, right=lead_R, Ef=0.0)
        
        if return_positions:
            return device, positions
        return device, None


class CustomSystemBuilder:
    """
    Builder for custom geometries with flexible site addition.
    """
    
    def __init__(self):
        """Initialize empty system."""
        self.sites: List[np.ndarray] = []
        self.hoppings: List[Tuple[int, int, complex]] = []
        self.onsite: Dict[int, complex] = {}
    
    def add_site(self, position: np.ndarray, onsite: complex = 0.0) -> int:
        """
        Add a site to the system.
        
        Parameters
        ----------
        position : ndarray
            Site position (2D or 3D)
        onsite : complex
            On-site energy
            
        Returns
        -------
        site_id : int
            Index of added site
        """
        site_id = len(self.sites)
        self.sites.append(np.array(position))
        if onsite != 0.0:
            self.onsite[site_id] = onsite
        return site_id
    
    def add_hopping(self, site1: int, site2: int, t: complex):
        """
        Add hopping between sites.
        
        Parameters
        ----------
        site1, site2 : int
            Site indices
        t : complex
            Hopping amplitude
        """
        self.hoppings.append((site1, site2, t))
    
    def add_hoppings_by_distance(self, cutoff: float, t: complex):
        """
        Add hoppings between sites within cutoff distance.
        
        Parameters
        ----------
        cutoff : float
            Maximum distance for hopping
        t : complex
            Hopping amplitude
        """
        n = len(self.sites)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.sites[i] - self.sites[j])
                if dist <= cutoff:
                    self.add_hopping(i, j, t)
    
    def build_device(self, lead_left: PeriodicLead, lead_right: PeriodicLead) -> Device:
        """
        Build device from added sites and hoppings.
        
        Parameters
        ----------
        lead_left, lead_right : PeriodicLead
            Lead definitions
            
        Returns
        -------
        device : Device
            Assembled device
        """
        n = len(self.sites)
        H = sp.lil_matrix((n, n), dtype=complex)
        
        # Add on-site energies
        for site_id, eps in self.onsite.items():
            H[site_id, site_id] = eps
        
        # Add hoppings
        for i, j, t in self.hoppings:
            H[i, j] = t
            H[j, i] = np.conj(t)
        
        H = H.tocsr()
        
        return Device(H=H, S=None, left=lead_left, right=lead_right, Ef=0.0)


def make_1d_chain(n_sites: int, t: float = 1.0, onsite: float = 0.0) -> Device:
    """
    Quick builder for 1D tight-binding chain.
    
    Parameters
    ----------
    n_sites : int
        Chain length
    t : float
        Hopping parameter
    onsite : float
        On-site energy
        
    Returns
    -------
    device : Device
        1D chain device
    """
    H = sp.diags([onsite], [0], shape=(n_sites, n_sites), dtype=complex)
    H += sp.diags([-t, -t], [-1, 1], shape=(n_sites, n_sites), dtype=complex)
    H = H.tocsr()
    
    H00 = np.array([[onsite]], dtype=complex)
    H01 = np.array([[-t]], dtype=complex)
    
    lead = PeriodicLead(H00=H00, H01=H01)
    
    return Device(H=H, S=None, left=lead, right=lead, Ef=0.0)


def info():
    """Print available builders."""
    print("ThoothQT System Builders")
    print("=" * 60)
    print()
    print("Available builders:")
    print("  • GrapheneBuilder - Zigzag and armchair nanoribbons")
    print("  • CustomSystemBuilder - Flexible geometry builder")
    print("  • make_1d_chain() - Quick 1D chain")
    print()
    print("Example:")
    print("  builder = GrapheneBuilder(a=1.42, t=2.7)")
    print("  device, pos = builder.zigzag_ribbon(width=10, length=20)")
    print()
