"""
Graphene Nanoribbon Builder v2.0
=================================
Proper honeycomb lattice implementation for zigzag and armchair ribbons.

This version correctly implements the graphene honeycomb structure with
proper unit cell definitions for NEGF transport calculations.
"""
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
from toothqt_production import Device, PeriodicLead


class GrapheneRibbonBuilder:
    """
    Build graphene nanoribbons with correct honeycomb geometry.
    
    The honeycomb lattice has two sublattices (A and B) with nearest-neighbor
    distance a = 1.42 Å (C-C bond length).
    
    Primitive vectors for graphene:
        a1 = a * (√3, 0)
        a2 = a * (√3/2, 3/2)
    
    For zigzag ribbons:
        - Transport direction: along x (a1 direction)
        - Width: number of zigzag chains (perpendicular to transport)
        - Unit cell: contains 2*width atoms (width from each sublattice)
    
    For armchair ribbons:
        - Transport direction: along y (a2 direction)  
        - Width: number of dimer lines (perpendicular to transport)
        - Unit cell: contains width*4 atoms (dimer structure)
    """
    
    def __init__(self, a: float = 1.42, t: float = -2.7):
        """
        Initialize graphene builder.
        
        Parameters
        ----------
        a : float
            Carbon-carbon bond length (Å)
        t : float
            Nearest-neighbor hopping energy (eV)
        """
        self.a = a
        self.t = t
        
        # Graphene primitive vectors (KWANT convention for zigzag transport)
        # Transport along x uses minimal rectangular unit cell
        self.a1 = np.array([a, 0])  # Transport direction (one C-C bond)
        self.a2 = np.array([a / 2, np.sqrt(3) * a / 2])  # Transverse direction
        
        # Sublattice positions within primitive cell
        self.r_A = np.array([0, 0])
        self.r_B = np.array([0, a / np.sqrt(3)])  # NOT [0, a]!
    
    def zigzag_ribbon(self, width: int, length: int, 
                     return_positions: bool = False) -> Tuple[Device, Optional[np.ndarray]]:
        """
        Build a zigzag graphene nanoribbon.
        
        For a zigzag ribbon, the transport is along the x-direction (a1).
        Each unit cell contains one zigzag "row" with width pairs of A/B atoms.
        
        Structure (width=2, viewed from above):
            A0--B0     A4--B4     A8--B8
              \/  \      \/  \      \/  \
              /\   \     /\   \     /\   \
            B1  A1--B5  A5--B9  A9--...
              \  /\      \  /\      \  /
               \/  \      \/  \      \/
            A2--B2  A6--B6  A10-B10
              \/  \  \/  \  \/  \
              /\   \/\   \/\   \
            B3  A3--B7  A7--B11 A11
        
        Parameters
        ----------
        width : int
            Number of zigzag chains (minimum 2)
        length : int
            Number of unit cells in transport direction
        return_positions : bool
            If True, return (device, positions), else (device, None)
            
        Returns
        -------
        device : Device
            ThoothQT device with leads
        positions : ndarray or None
            Atomic positions (n_atoms × 2) if requested
        """
        if width < 2:
            raise ValueError("Width must be >= 2 for zigzag ribbon")
        
        # Total atoms: length * 2*width
        n_atoms = length * 2 * width
        n_lead = 2 * width  # Atoms per unit cell
        
        # Initialize arrays
        positions = np.zeros((n_atoms, 2)) if return_positions else None
        H = sp.lil_matrix((n_atoms, n_atoms), dtype=complex)
        
        # Atom indexing: match KWANT's ordering
        # KWANT orders: A sublattice in REVERSE iy, then B sublattice in normal iy
        # For width=2: A1, A0, B0, B1
        def idx_A(ix, iy):
            """Index for A sublattice atom at cell ix, chain iy"""
            # A sublattice in reverse order: width-1-iy
            return ix * n_lead + (width - 1 - iy)
        
        def idx_B(ix, iy):
            """Index for B sublattice atom at cell ix, chain iy"""
            # B sublattice in normal order: width + iy
            return ix * n_lead + width + iy
        
        # Build device cell by cell
        # First, place all atoms using KWANT's position formula:
        # pos = ix * a1 + iy * a2 + r_sublattice
        atom_positions_all = {}
        for ix in range(length):
            for iy in range(width):
                i_A = idx_A(ix, iy)
                i_B = idx_B(ix, iy)
                
                pos_A = ix * self.a1 + iy * self.a2 + self.r_A
                pos_B = ix * self.a1 + iy * self.a2 + self.r_B
                
                atom_positions_all[i_A] = pos_A
                atom_positions_all[i_B] = pos_B
                
                if return_positions:
                    positions[i_A] = pos_A
                    positions[i_B] = pos_B
        
        # Now add bonds based on distance (nearest-neighbor only)
        # For graphene honeycomb, nearest neighbor distance is exactly a
        bond_tol_lower = 0.95 * self.a  # Lower bound
        bond_tol_upper = 1.05 * self.a  # Upper bound
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(atom_positions_all[i] - atom_positions_all[j])
                if bond_tol_lower < dist < bond_tol_upper:
                    H[i, j] = self.t
                    H[j, i] = self.t
        
        H = H.tocsr()
        
        # Build lead matrices (single unit cell)
        # Use distance-based approach with correct positions
        H00 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        H01 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        
        # Positions for one unit cell (ix=0) and next cell (ix=1)
        lead_pos = {}
        next_pos = {}
        for iy in range(width):
            i_A = width - 1 - iy  # A sublattice in reverse
            i_B = width + iy  # B sublattice normal
            lead_pos[i_A] = 0 * self.a1 + iy * self.a2 + self.r_A
            lead_pos[i_B] = 0 * self.a1 + iy * self.a2 + self.r_B
            
            next_pos[i_A] = 1 * self.a1 + iy * self.a2 + self.r_A
            next_pos[i_B] = 1 * self.a1 + iy * self.a2 + self.r_B
        
        bond_tol_lower = 0.95 * self.a
        bond_tol_upper = 1.05 * self.a
        
        # Intra-cell bonds (H00)
        for i in range(n_lead):
            for j in range(i + 1, n_lead):
                dist = np.linalg.norm(lead_pos[i] - lead_pos[j])
                if bond_tol_lower < dist < bond_tol_upper:
                    H00[i, j] = self.t
                    H00[j, i] = self.t
        
        # Inter-cell bonds (H01)
        for i in range(n_lead):
            for j in range(n_lead):
                dist = np.linalg.norm(lead_pos[i] - next_pos[j])
                if bond_tol_lower < dist < bond_tol_upper:
                    H01[i, j] = self.t
        
        H00 = H00.toarray()
        H01 = H01.toarray()
        
        # Create leads
        lead_L = PeriodicLead(H00=H00, H01=H01)
        lead_R = PeriodicLead(H00=H00, H01=H01)
        
        # Create device (Fermi level at Dirac point E_F = 0)
        device = Device(H=H, S=None, left=lead_L, right=lead_R, Ef=0.0)
        
        return device, positions
    
    def armchair_ribbon(self, width: int, length: int,
                       return_positions: bool = False) -> Tuple[Device, Optional[np.ndarray]]:
        """
        Build an armchair graphene nanoribbon.
        
        For an armchair ribbon, the transport is along the y-direction.
        Each unit cell contains 'width' dimer lines (4 atoms per dimer line).
        
        Structure (width=2, viewed from above):
            A--B  A--B    <- dimer line 0
            |  |  |  |
            B--A  B--A    <- dimer line 0
            
            A--B  A--B    <- dimer line 1
            |  |  |  |
            B--A  B--A    <- dimer line 1
        
        Parameters
        ----------
        width : int
            Number of dimer lines (minimum 1)
        length : int
            Number of unit cells in transport direction
        return_positions : bool
            If True, return (device, positions), else (device, None)
            
        Returns
        -------
        device : Device
            ThoothQT device with leads
        positions : ndarray or None
            Atomic positions (n_atoms × 2) if requested
        """
        if width < 1:
            raise ValueError("Width must be >= 1 for armchair ribbon")
        
        # Armchair unit cell: 4 atoms per dimer line
        n_lead = 4 * width
        n_atoms = length * n_lead
        
        # Initialize arrays
        positions = np.zeros((n_atoms, 2)) if return_positions else None
        H = sp.lil_matrix((n_atoms, n_atoms), dtype=complex)
        
        # Define positions for armchair structure
        # Each dimer line has 4 atoms in vertical arrangement
        dx = np.sqrt(3) * self.a / 2
        dy = 3 * self.a / 2
        
        # Build device
        for iy in range(length):
            y0 = iy * dy
            
            for iw in range(width):
                x0 = iw * 3 * self.a
                
                # Four atoms in this dimer line
                # Top dimer: A-B
                idx_A1 = iy * n_lead + iw * 4 + 0
                idx_B1 = iy * n_lead + iw * 4 + 1
                # Bottom dimer: B-A
                idx_B2 = iy * n_lead + iw * 4 + 2
                idx_A2 = iy * n_lead + iw * 4 + 3
                
                if return_positions:
                    positions[idx_A1] = [x0, y0]
                    positions[idx_B1] = [x0 + np.sqrt(3) * self.a, y0]
                    positions[idx_B2] = [x0 + dx, y0 + self.a]
                    positions[idx_A2] = [x0 + dx + np.sqrt(3) * self.a, y0 + self.a]
                
                # Horizontal bonds within dimer
                H[idx_A1, idx_B1] = self.t
                H[idx_B1, idx_A1] = self.t
                H[idx_B2, idx_A2] = self.t
                H[idx_A2, idx_B2] = self.t
                
                # Vertical bonds within unit cell
                H[idx_A1, idx_B2] = self.t
                H[idx_B2, idx_A1] = self.t
                H[idx_B1, idx_A2] = self.t
                H[idx_A2, idx_B1] = self.t
                
                # Horizontal bonds to next dimer line (within unit cell)
                if iw < width - 1:
                    # Connect to next dimer line
                    idx_A1_next = iy * n_lead + (iw + 1) * 4 + 0
                    idx_B2_next = iy * n_lead + (iw + 1) * 4 + 2
                    
                    H[idx_B1, idx_A1_next] = self.t
                    H[idx_A1_next, idx_B1] = self.t
                    H[idx_A2, idx_B2_next] = self.t
                    H[idx_B2_next, idx_A2] = self.t
                
                # Inter-cell bonds (to next unit cell along y)
                if iy < length - 1:
                    jdx_A1 = (iy + 1) * n_lead + iw * 4 + 0
                    jdx_B1 = (iy + 1) * n_lead + iw * 4 + 1
                    jdx_B2 = (iy + 1) * n_lead + iw * 4 + 2
                    jdx_A2 = (iy + 1) * n_lead + iw * 4 + 3
                    
                    # Bottom of current cell to top of next cell
                    H[idx_B2, jdx_A1] = self.t
                    H[jdx_A1, idx_B2] = self.t
                    H[idx_A2, jdx_B1] = self.t
                    H[jdx_B1, idx_A2] = self.t
        
        H = H.tocsr()
        
        # Build lead matrices
        H00 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        H01 = sp.lil_matrix((n_lead, n_lead), dtype=complex)
        
        for iw in range(width):
            # Indices for this dimer line
            i_A1 = iw * 4 + 0
            i_B1 = iw * 4 + 1
            i_B2 = iw * 4 + 2
            i_A2 = iw * 4 + 3
            
            # Intra-cell bonds (H00)
            H00[i_A1, i_B1] = self.t
            H00[i_B1, i_A1] = self.t
            H00[i_B2, i_A2] = self.t
            H00[i_A2, i_B2] = self.t
            H00[i_A1, i_B2] = self.t
            H00[i_B2, i_A1] = self.t
            H00[i_B1, i_A2] = self.t
            H00[i_A2, i_B1] = self.t
            
            # Bonds to next dimer line (still H00)
            if iw < width - 1:
                i_A1_next = (iw + 1) * 4 + 0
                i_B2_next = (iw + 1) * 4 + 2
                
                H00[i_B1, i_A1_next] = self.t
                H00[i_A1_next, i_B1] = self.t
                H00[i_A2, i_B2_next] = self.t
                H00[i_B2_next, i_A2] = self.t
            
            # Inter-cell bonds (H01)
            H01[i_B2, i_A1] = self.t
            H01[i_A2, i_B1] = self.t
        
        H00 = H00.toarray()
        H01 = H01.toarray()
        
        # Create leads
        lead_L = PeriodicLead(H00=H00, H01=H01)
        lead_R = PeriodicLead(H00=H00, H01=H01)
        
        # Create device
        device = Device(H=H, S=None, left=lead_L, right=lead_R, Ef=0.0)
        
        return device, positions


def validate_graphene_structure(device: Device, width: int, ribbon_type: str = 'zigzag'):
    """
    Validate that the graphene structure is correct.
    
    Checks:
    1. H00 and H01 have correct shape
    2. All matrices are Hermitian
    3. det(H00) and det(H01) are non-zero
    4. Device structure matches lead structure at boundaries
    5. Band structure is reasonable for graphene
    
    Parameters
    ----------
    device : Device
        Device to validate
    width : int
        Ribbon width
    ribbon_type : str
        'zigzag' or 'armchair'
    
    Returns
    -------
    valid : bool
        True if all checks pass
    report : str
        Validation report
    """
    report = []
    report.append("=" * 80)
    report.append(f"GRAPHENE {ribbon_type.upper()} RIBBON VALIDATION")
    report.append("=" * 80)
    
    H_device = device.H.toarray() if sp.issparse(device.H) else device.H
    H00 = device.left.H00
    H01 = device.left.H01
    
    n_device = H_device.shape[0]
    n_lead = H00.shape[0]
    
    report.append(f"\nDimensions:")
    report.append(f"  Device: {n_device} atoms")
    report.append(f"  Lead: {n_lead} atoms/cell")
    report.append(f"  Expected lead size: {2*width if ribbon_type=='zigzag' else 4*width}")
    
    # Check 1: Correct dimensions
    checks_passed = 0
    total_checks = 7
    
    expected_lead = 2 * width if ribbon_type == 'zigzag' else 4 * width
    if n_lead == expected_lead:
        report.append(f"  ✓ Lead size matches width")
        checks_passed += 1
    else:
        report.append(f"  ✗ Lead size mismatch!")
        
    # Check 2: Hermitian matrices
    if np.allclose(H_device, H_device.conj().T):
        report.append(f"  ✓ Device Hamiltonian is Hermitian")
        checks_passed += 1
    else:
        report.append(f"  ✗ Device Hamiltonian not Hermitian!")
        
    if np.allclose(H00, H00.conj().T):
        report.append(f"  ✓ H00 is Hermitian")
        checks_passed += 1
    else:
        report.append(f"  ✗ H00 not Hermitian!")
    
    # Check 3: Non-singular
    det_H00 = np.linalg.det(H00)
    det_H01 = np.linalg.det(H01)
    
    report.append(f"\nDeterminants:")
    report.append(f"  det(H00) = {det_H00:.6e}")
    report.append(f"  det(H01) = {det_H01:.6e}")
    
    if abs(det_H00) > 1e-10:
        report.append(f"  ✓ H00 is non-singular")
        checks_passed += 1
    else:
        report.append(f"  ✗ H00 is singular!")
        
    if abs(det_H01) > 1e-10:
        report.append(f"  ✓ H01 is non-singular")
        checks_passed += 1
    else:
        report.append(f"  ✗ H01 is singular!")
    
    # Check 4: Boundary matching
    H_first = H_device[:n_lead, :n_lead]
    H_coupling = H_device[:n_lead, n_lead:2*n_lead] if n_device >= 2*n_lead else None
    
    report.append(f"\nBoundary Structure:")
    if np.allclose(H_first, H00, atol=1e-10):
        report.append(f"  ✓ First device cell matches H00")
        checks_passed += 1
    else:
        report.append(f"  ✗ First device cell doesn't match H00!")
        report.append(f"    Max difference: {np.max(np.abs(H_first - H00)):.6e}")
    
    if H_coupling is not None:
        if np.allclose(H_coupling, H01, atol=1e-10):
            report.append(f"  ✓ Device coupling matches H01")
            checks_passed += 1
        else:
            report.append(f"  ✗ Device coupling doesn't match H01!")
            report.append(f"    Max difference: {np.max(np.abs(H_coupling - H01)):.6e}")
    else:
        report.append(f"  - Device too short to check coupling")
        total_checks -= 1
    
    # Check 5: Band structure
    report.append(f"\nBand Structure:")
    H_full = H00 + H01 + H01.conj().T
    eigs = np.linalg.eigvalsh(H_full)
    report.append(f"  Eigenvalues: [{eigs[0]:.3f}, {eigs[-1]:.3f}] eV")
    
    # For graphene, should have states near E=0 (Dirac point)
    if np.min(np.abs(eigs)) < 1.0:
        report.append(f"  ✓ Has states near Dirac point (E=0)")
    else:
        report.append(f"  ! No states near Dirac point (check structure)")
    
    # Summary
    report.append(f"\n{'=' * 80}")
    report.append(f"VALIDATION: {checks_passed}/{total_checks} checks passed")
    report.append(f"{'=' * 80}")
    
    return checks_passed == total_checks, "\n".join(report)


if __name__ == "__main__":
    # Test the new builder
    print("Testing GrapheneRibbonBuilder v2.0")
    print()
    
    builder = GrapheneRibbonBuilder(a=1.42, t=-2.7)
    
    # Test zigzag
    print("Building zigzag ribbon (width=2, length=3)...")
    device_zz, pos_zz = builder.zigzag_ribbon(width=2, length=3, return_positions=True)
    valid_zz, report_zz = validate_graphene_structure(device_zz, width=2, ribbon_type='zigzag')
    print(report_zz)
    
    print("\n" + "=" * 80)
    print()
    
    # Test armchair
    print("Building armchair ribbon (width=2, length=3)...")
    device_ac, pos_ac = builder.armchair_ribbon(width=2, length=3, return_positions=True)
    valid_ac, report_ac = validate_graphene_structure(device_ac, width=2, ribbon_type='armchair')
    print(report_ac)
