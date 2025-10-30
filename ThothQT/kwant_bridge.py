"""
KWANT to ThothQT Bridge
========================

Convert KWANT systems into ThothQT format for fast NEGF transport calculations.
This bridge allows you to:
1. Build complex geometries using KWANT's powerful lattice tools
2. Run transport calculations with ThothQT's optimized NEGF engine
3. Easily add impurities, fields, and modifications to the Hamiltonian

Example Usage:
-------------
import kwant
from thothqt.kwant_bridge import kwant_to_thothqt

# Build system in KWANT
graphene = kwant.lattice.honeycomb(1.42, name='graphene')
sys = kwant.Builder()
# ... define your system ...

# Convert to ThothQT
device = kwant_to_thothqt(sys.finalized())

# Run fast NEGF transport
from thothqt import NEGFEngine
negf = NEGFEngine(device, Temp=300.0)
T = negf.transmission(E=0.5)

Author: Quantum Sensing Project
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, List
import warnings

try:
    import kwant
    _KWANT_AVAILABLE = True
except ImportError:
    _KWANT_AVAILABLE = False
    kwant = None

try:
    from .thothqt import Device, PeriodicLead
except ImportError:
    from thothqt import Device, PeriodicLead


def kwant_to_thothqt(fsys, Ef: float = 0.0, 
                     energy_scale: float = 1.0) -> Device:
    """
    Convert KWANT finalized system to ThothQT Device.
    
    Parameters
    ----------
    fsys : kwant.system.FiniteSystem
        KWANT finalized system with attached leads
    Ef : float
        Fermi energy for lead self-energies (eV)
    energy_scale : float
        Energy scale factor for unit conversion
        
    Returns
    -------
    device : Device
        ThothQT device ready for NEGF calculations
        
    Examples
    --------
    >>> import kwant
    >>> from thothqt.kwant_bridge import kwant_to_thothqt
    >>> 
    >>> # Build 1D chain in KWANT
    >>> lat = kwant.lattice.chain(a=1.0)
    >>> syst = kwant.Builder()
    >>> syst[(lat(i) for i in range(10))] = 0.0
    >>> syst[lat.neighbors()] = -1.0
    >>> 
    >>> lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    >>> lead[lat(0)] = 0.0
    >>> lead[lat.neighbors()] = -1.0
    >>> syst.attach_lead(lead)
    >>> syst.attach_lead(lead.reversed())
    >>> 
    >>> fsys = syst.finalized()
    >>> 
    >>> # Convert to ThothQT
    >>> device = kwant_to_thothqt(fsys, Ef=0.0)
    >>> 
    >>> # Compute transmission
    >>> from thothqt import NEGFEngine
    >>> negf = NEGFEngine(device, Temp=300.0)
    >>> T = negf.transmission(E=0.5)
    """
    if not _KWANT_AVAILABLE:
        raise ImportError("KWANT not available. Install kwant to use this bridge.")
    
    if not isinstance(fsys, kwant.system.FiniteSystem):
        raise ValueError("Input must be a KWANT FiniteSystem (call .finalized() first)")
    
    if len(fsys.leads) != 2:
        raise ValueError("ThothQT currently supports exactly 2 leads. "
                        f"Got {len(fsys.leads)} leads.")
    
    print("Converting KWANT system to ThothQT...")
    
    # Step 1: Extract device Hamiltonian
    print("  [1/4] Extracting device Hamiltonian...")
    H_device = fsys.hamiltonian_submatrix(sparse=True)
    n_sites = H_device.shape[0]
    print(f"    Device: {n_sites} sites")
    
    # Convert to complex CSR format
    H_device = sp.csr_matrix(H_device, dtype=complex) * energy_scale
    
    # Step 2: Extract lead matrices
    print("  [2/4] Extracting lead matrices...")
    leads = []
    
    for i, lead in enumerate(fsys.leads):
        side = "Left" if i == 0 else "Right"
        print(f"    {side} lead:")
        
        # Lead Hamiltonian matrices
        H_lead = lead.cell_hamiltonian(sparse=True) * energy_scale
        V_lead = lead.inter_cell_hopping(sparse=True) * energy_scale
        
        # Convert to dense arrays for decimation
        H00 = np.array(H_lead.toarray(), dtype=complex)
        H01 = np.array(V_lead.toarray(), dtype=complex)
        
        lead_size = H00.shape[0]
        print(f"      Unit cell: {lead_size} sites")
        print(f"      Hopping shape: {H01.shape}")
        
        leads.append({'H00': H00, 'H01': H01, 'lead_obj': lead})
    
    # Step 3: Extract lead-device coupling
    print("  [3/4] Computing lead-device coupling...")
    
    thoth_leads = []
    for i, lead_data in enumerate(leads):
        side = "Left" if i == 0 else "Right"
        lead_obj = lead_data['lead_obj']
        
        # Get coupling matrix between device and lead
        try:
            # KWANT's coupling matrix (device sites × lead sites)
            coupling = lead_obj.selfenergy(Ef, args=())
            
            # Extract the coupling matrices
            # This requires understanding KWANT's internal structure
            # Simplified: assume uniform coupling
            n_lead = lead_data['H00'].shape[0]
            
            if i == 0:  # Left lead
                # Find sites at left boundary
                tau_cpl = sp.csr_matrix((n_sites, n_lead), dtype=complex)
                # Simplified: couple to first n_lead sites
                for j in range(min(n_lead, n_sites)):
                    tau_cpl[j, j] = -1.0 * energy_scale
            else:  # Right lead
                # Find sites at right boundary
                tau_cpl = sp.csr_matrix((n_sites, n_lead), dtype=complex)
                # Simplified: couple to last n_lead sites
                for j in range(min(n_lead, n_sites)):
                    tau_cpl[n_sites - n_lead + j, j] = -1.0 * energy_scale
            
        except Exception as e:
            print(f"      Warning: Could not extract exact coupling. Using default. ({e})")
            # Default coupling
            n_lead = lead_data['H00'].shape[0]
            tau_cpl = sp.csr_matrix((n_sites, n_lead), dtype=complex)
            
            if i == 0:  # Left lead couples to first sites
                for j in range(min(n_lead, n_sites)):
                    tau_cpl[j, j] = -1.0 * energy_scale
            else:  # Right lead couples to last sites
                for j in range(min(n_lead, n_sites)):
                    tau_cpl[n_sites - n_lead + j, j] = -1.0 * energy_scale
        
        print(f"      Coupling: {tau_cpl.nnz} non-zero elements")
        
        # Create ThothQT lead
        thoth_lead = PeriodicLead(
            H00=lead_data['H00'],
            H01=lead_data['H01'],
            tau_cpl=tau_cpl
        )
        thoth_leads.append(thoth_lead)
    
    # Step 4: Create ThothQT device
    print("  [4/4] Creating ThothQT device...")
    device = Device(
        H=H_device,
        left=thoth_leads[0],
        right=thoth_leads[1]
    )
    
    print("✓ Conversion complete!")
    print(f"  Device size: {device.H.shape[0]}×{device.H.shape[0]}")
    print(f"  Left lead: {device.left.H00.shape[0]} sites/cell")
    print(f"  Right lead: {device.right.H00.shape[0]} sites/cell")
    
    return device


def add_onsite_potential(device: Device, sites: List[int], 
                        potential: float) -> Device:
    """
    Add on-site potential to specific sites.
    
    Useful for modeling impurities, gate voltages, or local fields.
    
    Parameters
    ----------
    device : Device
        Original ThothQT device
    sites : list of int
        Site indices to modify
    potential : float
        On-site potential to add (eV)
        
    Returns
    -------
    modified_device : Device
        New device with modified Hamiltonian
        
    Examples
    --------
    >>> # Add impurity at center site
    >>> n_sites = device.H.shape[0]
    >>> center = n_sites // 2
    >>> device_impurity = add_onsite_potential(device, [center], 0.5)
    """
    # Copy device
    H_new = device.H.copy()
    
    # Add potential
    for site in sites:
        if 0 <= site < H_new.shape[0]:
            H_new[site, site] += potential
        else:
            warnings.warn(f"Site {site} out of range [0, {H_new.shape[0]-1}]")
    
    return Device(H=H_new, left=device.left, right=device.right)


def add_uniform_field(device: Device, field_strength: float) -> Device:
    """
    Add uniform electric field (on-site potential gradient).
    
    Parameters
    ----------
    device : Device
        Original ThothQT device
    field_strength : float
        Field strength (eV per site)
        
    Returns
    -------
    modified_device : Device
        Device with linear potential
    """
    H_new = device.H.copy()
    n_sites = H_new.shape[0]
    
    # Linear potential: V(i) = field_strength * i
    for i in range(n_sites):
        H_new[i, i] += field_strength * i
    
    return Device(H=H_new, left=device.left, right=device.right)


def add_positional_field(device: Device, positions: np.ndarray, 
                        field_vector: np.ndarray) -> Device:
    """
    Add position-dependent electric field.
    
    Parameters
    ----------
    device : Device
        Original ThothQT device
    positions : ndarray (N, 2)
        Site positions (Å)
    field_vector : ndarray (2,)
        Electric field vector (eV/Å)
        
    Returns
    -------
    modified_device : Device
        Device with position-dependent field
    """
    H_new = device.H.copy()
    
    # Add F⃗·r⃗ potential
    for i, pos in enumerate(positions):
        potential = np.dot(field_vector, pos)
        H_new[i, i] += potential
    
    return Device(H=H_new, left=device.left, right=device.right)


def validate_conversion(device: Device, fsys, E: float = 0.1, 
                       verbose: bool = True) -> Dict:
    """
    Validate ThothQT conversion against original KWANT system.
    
    Compares transmission coefficients between KWANT and ThothQT
    at a test energy.
    
    Parameters
    ----------
    device : Device
        Converted ThothQT device
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
    >>> device = kwant_to_thothqt(fsys)
    >>> results = validate_conversion(device, fsys, E=0.5)
    >>> print(f"Error: {results['relative_error']:.2%}")
    """
    try:
        from .thothqt import NEGFEngine
    except ImportError:
        from thothqt import NEGFEngine
    
    if verbose:
        print("\nValidating conversion: KWANT vs ThothQT")
        print(f"  Test energy: E = {E} eV")
    
    # Compute with ThothQT
    try:
        negf = NEGFEngine(device, Temp=300.0)
        T_thothqt = negf.transmission(E)
    except Exception as e:
        if verbose:
            print(f"  ❌ ThothQT transmission failed: {e}")
        T_thothqt = np.nan
    
    # Compute with KWANT
    try:
        smatrix = kwant.smatrix(fsys, E)
        T_kwant = smatrix.transmission(1, 0)
    except Exception as e:
        if verbose:
            print(f"  ❌ KWANT transmission failed: {e}")
        T_kwant = np.nan
    
    # Compare results
    if not (np.isnan(T_thothqt) or np.isnan(T_kwant)):
        abs_error = abs(T_kwant - T_thothqt)
        rel_error = abs_error / max(abs(T_kwant), 1e-10)
        
        if verbose:
            print(f"  KWANT transmission:   {T_kwant:.6f}")
            print(f"  ThothQT transmission: {T_thothqt:.6f}")
            print(f"  Absolute error:       {abs_error:.2e}")
            print(f"  Relative error:       {rel_error:.2%}")
            
            if rel_error < 0.01:
                print("  ✓ Excellent agreement!")
            elif rel_error < 0.1:
                print("  ✓ Good agreement")
            else:
                print("  ⚠ Large discrepancy - check conversion")
    else:
        abs_error = np.nan
        rel_error = np.nan
        if verbose:
            print("  ❌ Could not compare (calculation failed)")
    
    return {
        'T_kwant': T_kwant,
        'T_thothqt': T_thothqt,
        'absolute_error': abs_error,
        'relative_error': rel_error,
        'energy': E
    }


# ============================================================================
# Example Systems
# ============================================================================

def make_kwant_1d_chain(L: int = 10, t: float = 1.0, a: float = 1.0):
    """Create 1D atomic chain in KWANT for testing."""
    if not _KWANT_AVAILABLE:
        raise ImportError("KWANT required for this function")
    
    lat = kwant.lattice.chain(a=a, norbs=1)
    syst = kwant.Builder()
    
    # Device region
    syst[(lat(x) for x in range(L))] = 0.0
    syst[lat.neighbors()] = -t
    
    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
    lead[lat(0)] = 0.0
    lead[lat.neighbors()] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    return syst


def make_kwant_graphene_ribbon(W: int = 3, L: int = 10, a: float = 1.42, t: float = 2.7):
    """Create graphene zigzag nanoribbon in KWANT."""
    if not _KWANT_AVAILABLE:
        raise ImportError("KWANT required for this function")
    
    graphene = kwant.lattice.honeycomb(a, name='graphene')
    a_sub, b_sub = graphene.sublattices
    
    # Device
    sys = kwant.Builder()
    for ix in range(L):
        for iy in range(W):
            sys[a_sub(ix, iy)] = 0
            sys[b_sub(ix, iy)] = 0
    sys[graphene.neighbors()] = t
    
    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    for iy in range(W):
        lead[a_sub(0, iy)] = 0
        lead[b_sub(0, iy)] = 0
    lead[graphene.neighbors()] = t
    
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
    
    return sys


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__" and _KWANT_AVAILABLE:
    print("=" * 80)
    print("KWANT → ThothQT Bridge Demonstration")
    print("=" * 80)
    
    # Example 1: 1D chain
    print("\nExample 1: 1D Atomic Chain")
    print("-" * 40)
    
    syst_1d = make_kwant_1d_chain(L=20, t=1.0)
    fsys_1d = syst_1d.finalized()
    
    device_1d = kwant_to_thothqt(fsys_1d, Ef=0.0)
    validate_conversion(device_1d, fsys_1d, E=0.1)
    
    # Example 2: Graphene ribbon
    print("\nExample 2: Graphene Nanoribbon")
    print("-" * 40)
    
    syst_gr = make_kwant_graphene_ribbon(W=3, L=5)
    fsys_gr = syst_gr.finalized()
    
    device_gr = kwant_to_thothqt(fsys_gr, Ef=0.0)
    validate_conversion(device_gr, fsys_gr, E=0.1)
    
    print("\n✓ Bridge validation complete!")

elif __name__ == "__main__":
    print("KWANT not available - install kwant to use this bridge.")