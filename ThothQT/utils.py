"""
ThothQT Utilities
=================

Helper functions for quantum transport calculations:
- Fermi-Dirac distribution with overflow protection
- Energy mesh generation with adaptive refinement
- Physical constants and unit conversions
- Common analysis functions

Author: Quantum Sensing Project
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable
import warnings

# Physical constants (SI units)
KB_SI = 1.380649e-23        # Boltzmann constant (J/K)
KB_EV = 8.617333262145e-5   # Boltzmann constant (eV/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
H_PLANCK = 6.62607015e-34   # Planck constant (J·s)
HBAR = H_PLANCK / (2 * np.pi)  # Reduced Planck constant (J·s)

# Derived constants
G0_SI = 2 * E_CHARGE**2 / H_PLANCK  # Conductance quantum (S)
G0_INV = H_PLANCK / (2 * E_CHARGE**2)  # Resistance quantum (Ω)

# Optional GPU support
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def fermi_dirac(E: Union[float, np.ndarray], mu: float, kT: float, 
                gpu: bool = False) -> Union[float, np.ndarray]:
    """
    Fermi-Dirac distribution function with overflow protection.
    
    f(E) = 1 / (1 + exp((E - μ) / kT))
    
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
        
    Notes
    -----
    Uses overflow protection to avoid numerical issues:
    - For x > 50: f ≈ 0 (exp(50) ~ 5×10²¹)
    - For x < -50: f ≈ 1 (exp(-50) ~ 2×10⁻²²)
    
    Examples
    --------
    >>> E = np.linspace(-1, 1, 100)
    >>> f = fermi_dirac(E, mu=0.0, kT=0.026)  # Room temp
    >>> print(f"f(μ) = {fermi_dirac(0.0, 0.0, 0.026):.3f}")  # Should be 0.5
    """
    # Ensure kT > 0
    kT = max(kT, 1e-12)
    x = (E - mu) / kT
    
    # Choose array library
    if gpu and _CUPY_AVAILABLE:
        xp = cp
        x = cp.asarray(x)
    else:
        xp = np
        x = np.asarray(x)
    
    # Overflow protection
    out = xp.empty_like(x, dtype=float)
    
    # Masks for different regimes
    pos = x > 50   # Exponentially suppressed
    neg = x < -50  # Nearly filled
    mid = ~(pos | neg)  # Normal regime
    
    # Apply appropriate approximations
    out[pos] = 0.0
    out[neg] = 1.0
    if xp.any(mid):
        out[mid] = 1.0 / (1.0 + xp.exp(x[mid]))
    
    # Convert back from GPU if needed
    if gpu and _CUPY_AVAILABLE:
        out = cp.asnumpy(out)
    
    return float(out) if np.isscalar(E) else out


def derivative_fermi(E: Union[float, np.ndarray], mu: float, kT: float) -> Union[float, np.ndarray]:
    """
    Derivative of Fermi-Dirac distribution: df/dE = -(1/kT) × f × (1-f)
    
    This function appears in many transport formulas and thermal broadening effects.
    
    Parameters
    ----------
    E : float or array
        Energy values (eV)
    mu : float
        Chemical potential (eV)
    kT : float
        Thermal energy (eV)
        
    Returns
    -------
    df_dE : float or array
        Derivative df/dE
    """
    f = fermi_dirac(E, mu, kT)
    return -(1.0 / kT) * f * (1.0 - f)


def thermal_broadening_width(kT: float, factor: float = 5.0) -> float:
    """
    Estimate thermal broadening width for energy integration.
    
    Parameters
    ----------
    kT : float
        Thermal energy (eV)
    factor : float
        Number of thermal scales (default: 5kT covers ~99.3% of distribution)
        
    Returns
    -------
    width : float
        Energy width for integration (eV)
    """
    return factor * kT


class EnergyMesh:
    """
    Energy grid with optional adaptive refinement around specific points.
    
    Useful for adaptive integration when transmission or DOS varies rapidly
    near resonances, band edges, or impurity levels.
    """
    
    def __init__(self, Emin: float, Emax: float, n_base: int,
                 refine_at: Tuple[float, ...] = (), 
                 refine_width: float = 0.1, refine_points: int = 50):
        """
        Create energy mesh with optional refinement.
        
        Parameters
        ----------
        Emin, Emax : float
            Energy range (eV)
        n_base : int
            Number of points in base grid
        refine_at : tuple of float
            Energy points requiring refinement (eV)
        refine_width : float
            Half-width of refinement regions (eV)
        refine_points : int
            Number of points in each refinement region
            
        Examples
        --------
        >>> # Basic uniform grid
        >>> mesh = EnergyMesh(-2.0, 2.0, n_base=100)
        >>> 
        >>> # Grid with refinement at Fermi level
        >>> mesh = EnergyMesh(-2.0, 2.0, n_base=100, refine_at=(0.0,))
        >>> 
        >>> # Multiple refinement points
        >>> mesh = EnergyMesh(-5.0, 5.0, n_base=200, 
        ...                   refine_at=(-1.0, 0.0, 1.0), refine_width=0.2)
        """
        self.Emin = Emin
        self.Emax = Emax
        self.n_base = n_base
        
        # Base uniform grid
        base_grid = np.linspace(Emin, Emax, n_base)
        
        # Add refinement regions
        refined_grids = [base_grid]
        
        for E_center in refine_at:
            E_ref_min = max(Emin, E_center - refine_width)
            E_ref_max = min(Emax, E_center + refine_width)
            
            if E_ref_max > E_ref_min:
                refined_grids.append(np.linspace(E_ref_min, E_ref_max, refine_points))
        
        # Combine and remove duplicates
        if len(refined_grids) > 1:
            all_points = np.concatenate(refined_grids)
            # Remove duplicates with tolerance
            self.grid = np.unique(np.round(all_points / 1e-12) * 1e-12)
        else:
            self.grid = base_grid
            
        # Sort to ensure monotonic ordering
        self.grid.sort()
    
    def __len__(self) -> int:
        """Number of energy points."""
        return len(self.grid)
    
    def __getitem__(self, idx) -> float:
        """Get energy point by index."""
        return self.grid[idx]
    
    def __iter__(self):
        """Iterate over energy points."""
        return iter(self.grid)
    
    @property
    def energies(self) -> np.ndarray:
        """Energy array."""
        return self.grid
    
    @property 
    def spacing(self) -> np.ndarray:
        """Energy spacing dE for integration."""
        return np.diff(self.grid)
    
    def integrate(self, values: np.ndarray) -> float:
        """
        Trapezoidal integration over the mesh.
        
        Parameters
        ----------
        values : ndarray
            Function values at mesh points
            
        Returns
        -------
        integral : float
            Integrated value
        """
        if len(values) != len(self.grid):
            raise ValueError(f"Values array length {len(values)} != grid length {len(self.grid)}")
        
        return np.trapz(values, self.grid)


def conductance_to_si(G_quantum_units: float, temperature: float = 300.0) -> float:
    """
    Convert conductance from quantum units (2e²/h) to SI units (S).
    
    Parameters
    ----------
    G_quantum_units : float
        Conductance in units of 2e²/h
    temperature : float
        Temperature (K) - not used, for future extensions
        
    Returns
    -------
    G_si : float
        Conductance in Siemens (S)
        
    Examples
    --------
    >>> G_quantum = 1.0  # One quantum of conductance
    >>> G_si = conductance_to_si(G_quantum)
    >>> print(f"G₀ = {G_si:.2e} S")
    """
    return G_quantum_units * G0_SI


def resistance_to_si(R_quantum_units: float) -> float:
    """
    Convert resistance from quantum units (h/2e²) to SI units (Ω).
    
    Parameters
    ----------
    R_quantum_units : float
        Resistance in units of h/2e²
        
    Returns
    -------
    R_si : float
        Resistance in Ohms (Ω)
    """
    return R_quantum_units * G0_INV


def current_to_si(I_quantum_units: float, voltage: float) -> float:
    """
    Convert current from quantum units to SI (Amperes).
    
    In quantum transport: I = (2e²/h) × V × T(E)
    
    Parameters
    ----------
    I_quantum_units : float
        Current in quantum units (2e²/h × V)
    voltage : float
        Applied voltage (V)
        
    Returns
    -------
    I_si : float
        Current in Amperes (A)
    """
    return I_quantum_units * G0_SI * voltage


def temperature_to_thermal_energy(T_kelvin: float) -> float:
    """
    Convert temperature to thermal energy kT.
    
    Parameters
    ----------
    T_kelvin : float
        Temperature (K)
        
    Returns
    -------
    kT : float
        Thermal energy (eV)
        
    Examples
    --------
    >>> kT_room = temperature_to_thermal_energy(300.0)
    >>> print(f"Room temperature: kT = {kT_room:.3f} eV")
    """
    return KB_EV * T_kelvin


def thermal_energy_to_temperature(kT_eV: float) -> float:
    """
    Convert thermal energy to temperature.
    
    Parameters
    ----------
    kT_eV : float
        Thermal energy (eV)
        
    Returns
    -------
    T_kelvin : float
        Temperature (K)
    """
    return kT_eV / KB_EV


def quantum_of_conductance() -> float:
    """
    Return the quantum of conductance G₀ = 2e²/h in SI units.
    
    Returns
    -------
    G0 : float
        Quantum of conductance (S)
    """
    return G0_SI


def von_klitzing_constant() -> float:
    """
    Return the von Klitzing constant R_K = h/e² in SI units.
    
    Used in quantum Hall effect.
    
    Returns
    -------
    R_K : float
        von Klitzing constant (Ω)
    """
    return H_PLANCK / E_CHARGE**2


# ============================================================================
# Analysis Functions
# ============================================================================

def find_transmission_peaks(energies: np.ndarray, transmission: np.ndarray,
                           prominence: float = 0.1, width_min: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in transmission spectrum.
    
    Useful for identifying resonances in quantum systems.
    
    Parameters
    ----------
    energies : ndarray
        Energy values (eV)
    transmission : ndarray
        Transmission values
    prominence : float
        Minimum peak prominence
    width_min : int
        Minimum peak width (points)
        
    Returns
    -------
    peak_energies : ndarray
        Energies of identified peaks (eV)
    peak_heights : ndarray
        Heights of identified peaks
    """
    try:
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(transmission, 
                                     prominence=prominence,
                                     width=width_min)
        
        peak_energies = energies[peaks]
        peak_heights = transmission[peaks]
        
        return peak_energies, peak_heights
        
    except ImportError:
        warnings.warn("scipy not available for peak finding")
        return np.array([]), np.array([])


def compute_differential_conductance(energies: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    """
    Compute differential conductance dG/dE from transmission.
    
    Parameters
    ----------
    energies : ndarray
        Energy values (eV)
    transmission : ndarray
        Transmission values
        
    Returns
    -------
    dG_dE : ndarray
        Differential conductance (units of 2e²/h per eV)
    """
    return np.gradient(transmission, energies)


def compute_shot_noise(transmission: np.ndarray, current: float) -> float:
    """
    Compute shot noise power using Landauer formula.
    
    S = (2e²/h) × V × ∫ T(E) × (1 - T(E)) × (-df/dE) dE
    
    For low temperature: S ≈ 2eI × (1 - T) for single-mode limit
    
    Parameters
    ----------
    transmission : ndarray
        Transmission coefficients
    current : float
        DC current (quantum units)
        
    Returns
    -------
    shot_noise : float
        Shot noise power (quantum units)
        
    Notes
    -----
    This is a simplified calculation. Full shot noise requires
    integration over energy with proper Fermi function derivatives.
    """
    # Simplified: assume single mode and use average transmission
    T_avg = np.mean(transmission)
    return 2.0 * current * (1.0 - T_avg)  # Quantum units


def fano_factor(transmission: np.ndarray) -> float:
    """
    Compute Fano factor F = S/(2eI) for shot noise suppression.
    
    Parameters
    ----------
    transmission : ndarray
        Transmission coefficients
        
    Returns
    -------
    F : float
        Fano factor (F = 1 for Poissonian noise, F < 1 for sub-Poissonian)
    """
    T_avg = np.mean(transmission)
    if T_avg > 0:
        return 1.0 - T_avg  # For single mode
    else:
        return 0.0


# ============================================================================
# Convenience Functions
# ============================================================================

def room_temperature_kt() -> float:
    """Return thermal energy at room temperature (300 K)."""
    return temperature_to_thermal_energy(300.0)


def liquid_nitrogen_kt() -> float:
    """Return thermal energy at liquid nitrogen temperature (77 K)."""
    return temperature_to_thermal_energy(77.0)


def liquid_helium_kt() -> float:
    """Return thermal energy at liquid helium temperature (4.2 K)."""
    return temperature_to_thermal_energy(4.2)


def millikelvin_kt(T_mK: float = 10.0) -> float:
    """Return thermal energy at millikelvin temperature."""
    return temperature_to_thermal_energy(T_mK / 1000.0)


def create_bias_mesh(V_max: float, n_points: int = 51, 
                     symmetric: bool = True) -> np.ndarray:
    """
    Create bias voltage mesh for I-V curves.
    
    Parameters
    ----------
    V_max : float
        Maximum bias voltage (V)
    n_points : int
        Number of voltage points
    symmetric : bool
        If True, create symmetric range [-V_max, V_max]
        
    Returns
    -------
    voltages : ndarray
        Bias voltage array (V)
    """
    if symmetric:
        return np.linspace(-V_max, V_max, n_points)
    else:
        return np.linspace(0, V_max, n_points)


def create_temperature_series(T_min: float = 4.0, T_max: float = 300.0, 
                             n_points: int = 20, log_scale: bool = True) -> np.ndarray:
    """
    Create temperature series for thermal studies.
    
    Parameters
    ----------
    T_min, T_max : float
        Temperature range (K)
    n_points : int
        Number of temperature points
    log_scale : bool
        Use logarithmic spacing
        
    Returns
    -------
    temperatures : ndarray
        Temperature array (K)
    """
    if log_scale:
        return np.logspace(np.log10(T_min), np.log10(T_max), n_points)
    else:
        return np.linspace(T_min, T_max, n_points)


# ============================================================================
# Information Functions
# ============================================================================

def print_constants():
    """Print useful physical constants."""
    print("Physical Constants")
    print("=" * 50)
    print(f"Boltzmann constant:       kB = {KB_EV:.2e} eV/K")
    print(f"Elementary charge:        e  = {E_CHARGE:.2e} C")
    print(f"Planck constant:          h  = {H_PLANCK:.2e} J·s")
    print(f"Reduced Planck constant:  ℏ  = {HBAR:.2e} J·s")
    print(f"Quantum of conductance:   G₀ = {G0_SI:.2e} S")
    print(f"Quantum of resistance:    R₀ = {G0_INV:.2e} Ω")
    print(f"von Klitzing constant:    RK = {von_klitzing_constant():.2e} Ω")
    print()
    print("Useful Conversions")
    print("=" * 50)
    print(f"Room temperature (300 K): kT = {room_temperature_kt():.3f} eV")
    print(f"Liquid N₂ (77 K):        kT = {liquid_nitrogen_kt():.4f} eV")
    print(f"Liquid He (4.2 K):       kT = {liquid_helium_kt():.5f} eV")
    print(f"Millikelvin (10 mK):     kT = {millikelvin_kt(10):.6f} eV")


if __name__ == "__main__":
    print("=" * 70)
    print("ThothQT Utilities")
    print("=" * 70)
    print()
    
    print_constants()
    
    print("\nExample: Fermi function at room temperature")
    print("-" * 50)
    
    E = np.linspace(-0.2, 0.2, 100)
    kT = room_temperature_kt()
    f = fermi_dirac(E, mu=0.0, kT=kT)
    
    print(f"Thermal energy: kT = {kT:.3f} eV")
    print(f"f(μ-3kT) = {fermi_dirac(-3*kT, 0.0, kT):.3f}")
    print(f"f(μ) = {fermi_dirac(0.0, 0.0, kT):.3f}")
    print(f"f(μ+3kT) = {fermi_dirac(3*kT, 0.0, kT):.3f}")
    
    print("\n✓ Utilities module ready!")