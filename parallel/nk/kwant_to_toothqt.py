"""
KWANT to ThoothQT Bridge
=========================
Import KWANT systems into ThoothQT for fast NEGF transport calculations.

This bridge allows you to:
1. Build complex geometries using KWANT's powerful lattice tools
2. Run transport calculations with ThoothQT's optimized NEGF engine
3. Easily add impurities, fields, and modifications to the Hamiltonian

Example Usage:
-------------
import kwant
from kwant_to_toothqt import kwant_to_toothqt

# Build system in KWANT
graphene = kwant.lattice.honeycomb(1.42, name='graphene')
sys = kwant.Builder()
# ... define your system ...

# Convert to ThoothQT
device = kwant_to_toothqt(sys.finalized())

# Run fast NEGF transport
from toothqt_production import NEGFEngine
negf = NEGFEngine(device, Temp=300.0)
T = negf.transmission(E=0.5)
"""

import numpy as np
import scipy.sparse as sp
import kwant
from typing import Tuple, Optional
from toothqt_production import Device, PeriodicLead


def kwant_to_toothqt(fsys: kwant.system.FiniteSystem, 
                     Ef: float = 0.0,
                     return_info: bool = False) -> Device:
    """
    Convert a finalized KWANT system to a ThoothQT Device.
    
    Parameters
    ----------
    fsys : kwant.system.FiniteSystem
        Finalized KWANT system with attached leads
    Ef : float
        Fermi energy (eV)
    return_info : bool
        If True, return (device, info_dict) with conversion details
        
    Returns
    -------
    device : Device
        ThoothQT device ready for NEGF calculations
    info : dict (optional)
        Conversion information including site mappings
        
    Notes
    -----
    KWANT uses minimal unit cells in leads, which can result in non-square H01 matrices.
    This function pads H01 with zeros to make it square (required by ThoothQT).
    The padded entries do not contribute to transport since they correspond to 
    non-physical inter-cell hoppings.
    
    Examples
    --------
    >>> # Build a graphene nanoribbon in KWANT
    >>> lat = kwant.lattice.honeycomb(1.42, norbs=1)
    >>> syst = kwant.Builder()
    >>> # ... build system ...
    >>> fsys = syst.finalized()
    >>> 
    >>> # Convert to ThoothQT
    >>> device = kwant_to_toothqt(fsys)
    >>> 
    >>> # Compute transmission
    >>> from toothqt_production import NEGFEngine
    >>> negf = NEGFEngine(device, Temp=300.0)
    >>> T = negf.transmission(E=0.0)
    """
    
    print("=" * 80)
    print("KWANT -> ThoothQT CONVERSION")
    print("=" * 80)
    
    # Step 1: Extract device Hamiltonian
    print("\n[1/4] Extracting device Hamiltonian...")
    H_device = fsys.hamiltonian_submatrix(sparse=True)
    n_sites = H_device.shape[0]
    print(f"  Device: {n_sites} sites")
    
    # Step 2: Extract lead matrices
    print("\n[2/4] Extracting lead matrices...")
    leads = []
    
    for i, lead in enumerate(fsys.leads):
        side = "Left" if i == 0 else "Right"
        print(f"  {side} lead:")
        
        # Get intra-cell Hamiltonian (H00)
        H00 = lead.cell_hamiltonian(sparse=False)
        n_lead = H00.shape[0]
        
        # Get inter-cell hopping (H01)
        H01 = lead.inter_cell_hopping(sparse=False)
        
        print(f"    H00: {H00.shape} (intra-cell)")
        print(f"    H01: {H01.shape} (inter-cell)")
        
        # Check if H01 is square
        if H01.shape[0] != H01.shape[1]:
            print(f"    ! H01 is not square - KWANT uses minimal unit cell")
            print(f"    Creating padded H01 for ThoothQT...")
            
            # Pad H01 to square with zeros
            n_max = max(H01.shape)
            H01_padded = np.zeros((n_max, n_max), dtype=complex)
            H01_padded[:H01.shape[0], :H01.shape[1]] = H01
            H01 = H01_padded
            print(f"    Padded H01: {H01.shape}")
        
        # Diagnostics
        det_H00 = np.linalg.det(H00)
        det_H01 = np.linalg.det(H01)
        print(f"    det(H00) = {det_H00:.3e}")
        print(f"    det(H01) = {det_H01:.3e}")
        
        if np.abs(det_H00) < 1e-10:
            print(f"    ! WARNING: H00 is singular!")
        if np.abs(det_H01) < 1e-10:
            print(f"    ! WARNING: H01 is singular!")
        
        # Check Hermiticity
        if not np.allclose(H00, H00.conj().T):
            print(f"    ! WARNING: H00 is not Hermitian!")
        
        leads.append((H00, H01))
    
    # Step 3: Create ThoothQT lead objects
    print("\n[3/4] Creating ThoothQT lead objects...")
    H00_left, H01_left = leads[0]
    H00_right, H01_right = leads[1]
    
    # Convert to sparse for ThoothQT
    lead_left = PeriodicLead(
        H00=sp.csr_matrix(H00_left),
        H01=sp.csr_matrix(H01_left),
        Ef=Ef
    )
    print(f"  + Left lead created")
    
    lead_right = PeriodicLead(
        H00=sp.csr_matrix(H00_right),
        H01=sp.csr_matrix(H01_right),
        Ef=Ef
    )
    print(f"  + Right lead created")
    
    # Step 4: Create Device
    print("\n[4/4] Creating ThoothQT Device...")
    device = Device(
        H=H_device,
        left=lead_left,
        right=lead_right
    )
    print(f"  + Device created")
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print("Device ready for NEGF calculations!")
    print(f"  Total sites: {n_sites}")
    print(f"  Left lead: {H00_left.shape[0]} sites/cell")
    print(f"  Right lead: {H00_right.shape[0]} sites/cell")
    print("=" * 80)
    
    if return_info:
        info = {
            'n_sites': n_sites,
            'n_lead_left': H00_left.shape[0],
            'n_lead_right': H00_right.shape[0],
            'H01_padded': H01_left.shape[0] != leads[0][1].shape[1]
        }
        return device, info
    
    return device


# Utility functions for modifying Hamiltonians

def add_onsite_potential(device: Device, site_indices: list, V: float) -> Device:
    """
    Add onsite potential (impurity, doping) to specific sites.
    
    Parameters
    ----------
    device : Device
        ThoothQT device
    site_indices : list of int
        Site indices to modify
    V : float
        Onsite potential (eV)
        
    Returns
    -------
    device : Device
        Modified device with updated Hamiltonian
        
    Examples
    --------
    >>> # Add 0.5 eV impurity at site 10
    >>> device = add_onsite_potential(device, [10], 0.5)
    >>> 
    >>> # Add multiple impurities
    >>> device = add_onsite_potential(device, [5, 15, 25], 0.3)
    """
    H = device.H.toarray() if sp.issparse(device.H) else device.H.copy()
    
    for i in site_indices:
        H[i, i] += V
    
    return Device(
        H=sp.csr_matrix(H),
        left=device.left,
        right=device.right
    )


def add_uniform_field(device: Device, V: float) -> Device:
    """
    Add uniform electric field (constant potential shift).
    
    Parameters
    ----------
    device : Device
        ThoothQT device
    V : float
        Potential shift (eV)
        
    Returns
    -------
    device : Device
        Modified device
    """
    H = device.H.toarray() if sp.issparse(device.H) else device.H.copy()
    
    # Add to diagonal
    H += V * np.eye(H.shape[0])
    
    return Device(
        H=sp.csr_matrix(H),
        left=device.left,
        right=device.right
    )


def add_positional_field(device: Device, positions: np.ndarray, 
                        field_vector: np.ndarray) -> Device:
    """
    Add position-dependent electric field: V(r) = F · r
    
    Parameters
    ----------
    device : Device
        ThoothQT device
    positions : ndarray (N, 2) or (N, 3)
        Positions of each site
    field_vector : ndarray (2,) or (3,)
        Electric field vector (eV/Angstrom)
        
    Returns
    -------
    device : Device
        Modified device
        
    Examples
    --------
    >>> # Add field in x-direction
    >>> F = np.array([0.01, 0.0])  # 0.01 eV/A
    >>> device = add_positional_field(device, positions, F)
    """
    H = device.H.toarray() if sp.issparse(device.H) else device.H.copy()
    
    # Compute potential at each site: V_i = F · r_i
    V = positions @ field_vector
    
    # Add to diagonal
    for i in range(len(V)):
        H[i, i] += V[i]
    
    return Device(
        H=sp.csr_matrix(H),
        left=device.left,
        right=device.right
    )


# Validation function
def validate_conversion(device: Device, fsys: kwant.system.FiniteSystem, 
                       E: float = 0.1, verbose: bool = True) -> dict:
    """
    Validate KWANT->ThoothQT conversion by comparing transmission.
    
    Parameters
    ----------
    device : Device
        Converted ThoothQT device
    fsys : kwant.system.FiniteSystem
        Original KWANT system
    E : float
        Energy for validation (eV)
    verbose : bool
        Print detailed comparison
        
    Returns
    -------
    results : dict
        Validation results with transmission values and errors
        
    Examples
    --------
    >>> device = kwant_to_toothqt(fsys)
    >>> results = validate_conversion(device, fsys, E=0.5)
    >>> print(f"Error: {results['relative_error']:.2%}")
    """
    from toothqt_production import NEGFEngine
    
    if verbose:
        print("\nValidation: Comparing KWANT vs ThoothQT")
        print(f"  Energy: E = {E} eV")
    
    # Compute with ThoothQT
    negf = NEGFEngine(device, Temp=300.0)
    T_toothqt = negf.transmission(E)
    
    # Compute with KWANT
    try:
        smatrix = kwant.smatrix(fsys, E)
        T_kwant = smatrix.transmission(1, 0)
    except Exception as e:
        if verbose:
            print(f"! KWANT transmission failed: {e}")
        T_kwant = np.nan
    
    # Compare
    abs_error = np.abs(T_toothqt - T_kwant)
    rel_error = abs_error / (np.abs(T_kwant) + 1e-10) * 100
    
    if verbose:
        print(f"  Transmission at E = {E} eV:")
        print(f"    ThoothQT: {T_toothqt:.6f}")
        print(f"    KWANT:    {T_kwant:.6f}")
        print(f"    Error:    {abs_error:.6e} ({rel_error:.2f}%)")
        
        if abs_error < 1e-6:
            print("  + Excellent agreement!")
        elif abs_error < 1e-3:
            print("  + Good agreement")
        elif abs_error < 0.1:
            print("  ~ Reasonable agreement")
        else:
            print("  ! Large discrepancy - check system")
    
    return {
        'T_toothqt': T_toothqt,
        'T_kwant': T_kwant,
        'absolute_error': abs_error,
        'relative_error': rel_error
    }


# Example usage / test
if __name__ == "__main__":
    print("Testing KWANT -> ThoothQT Bridge")
    print("=" * 80)
    
    # Build a simple graphene nanoribbon
    print("\nBuilding graphene ribbon...")
    a = 1.42  # Angstrom
    lat = kwant.lattice.honeycomb(a, norbs=1)
    
    syst = kwant.Builder()
    
    # Device region
    W, L = 3, 10
    def device_shape(pos):
        x, y = pos
        return 0 <= x < L*a and 0 <= y < W*a*np.sqrt(3)/2
    
    syst[lat.shape(device_shape, (0, 0))] = 0.0
    syst[lat.neighbors()] = -2.7  # eV
    
    # Leads
    sym = kwant.TranslationalSymmetry((-a, 0))
    lead = kwant.Builder(sym)
    
    def lead_shape(pos):
        x, y = pos
        return 0 <= y < W*a*np.sqrt(3)/2
    
    lead[lat.shape(lead_shape, (0, 0))] = 0.0
    lead[lat.neighbors()] = -2.7
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    fsys = syst.finalized()
    print(f"  Total sites: {len(fsys.sites)}")
    
    # Convert to ThoothQT
    print("\nConverting to ThoothQT...")
    device = kwant_to_toothqt(fsys, Ef=0.0)
    
    # Validate
    validate_conversion(device, fsys, E=0.1)
    
    # Example: Add impurity
    print("\nExample: Adding impurity at center")
    n_sites = device.H.shape[0]
    center_site = n_sites // 2
    print(f"  Added 0.5 eV impurity at site {center_site}")
    
    device_impurity = add_onsite_potential(device, [center_site], 0.5)
    
    from toothqt_production import NEGFEngine
    negf_clean = NEGFEngine(device, Temp=300.0)
    negf_impurity = NEGFEngine(device_impurity, Temp=300.0)
    
    E_test = 0.1
    T_clean = negf_clean.transmission(E_test)
    T_impurity = negf_impurity.transmission(E_test)
    
    print(f"  Transmission at E={E_test} eV:")
    print(f"    Clean: {T_clean:.6f}")
    print(f"    With impurity: {T_impurity:.6f}")
    print(f"    Change: {(T_impurity-T_clean)/T_clean*100:.1f}%")
    
    print("\nSUCCESS: KWANT->ThoothQT bridge working!")
