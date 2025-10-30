"""
Self-Consistent Field (SCF) framework for quantum transport calculations.

This module provides SCF loops that iterate between NEGF transport calculations
and Poisson equation solving to achieve self-consistent electrostatic potentials.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
import warnings
from negf_core import NEGFSolver, extract_kwant_matrices
from poisson_solver import PoissonSolverGraphene
try:
    import cpp_negf  # optional C++ backend
    CPP_NEGF_AVAILABLE = True
except Exception:
    CPP_NEGF_AVAILABLE = False

class SCFSolver:
    """
    Self-consistent field solver for quantum transport with electrostatics.
    
    Iterates between:
    1. NEGF calculation -> charge density
    2. Poisson solver -> electrostatic potential  
    3. Update Hamiltonian with new potential
    4. Check convergence
    """
    def __init__(self, 
                 lattice_sites: np.ndarray,
                 epsilon: float = 11.7,
                 max_iterations: int = 100,
                 tolerance: float = 1e-9,
                 mixing_parameter: float = 0.1):  # More conservative mixing
        self.lattice_sites = np.asarray(lattice_sites)
        self.n_sites = len(lattice_sites)
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha = mixing_parameter
        
        # Initialize Poisson solver
        self.poisson_solver = PoissonSolverGraphene(lattice_sites, epsilon)
        
        # SCF history
        self.iteration_count = 0
        self.convergence_history = []
        self.potential_history = []
        self.charge_history = []
        self.current_history = []
        
    def linear_mixing(self, phi_new: np.ndarray, phi_old: np.ndarray) -> np.ndarray:
        return self.alpha * phi_new + (1.0 - self.alpha) * phi_old
    
    def anderson_mixing(self, phi_new: np.ndarray, phi_old: np.ndarray, 
                       history_size: int = 3) -> np.ndarray:
        # For first few iterations or if history is insufficient, use linear mixing
        if len(self.potential_history) < 2:
            return self.linear_mixing(phi_new, phi_old)
        
        try:
            # Simple Anderson mixing with last few iterations
            n_hist = min(len(self.potential_history), history_size)
            if n_hist < 2:
                return self.linear_mixing(phi_new, phi_old)
            
            # Get residuals: R_i = phi_{i+1} - phi_i
            residuals = []
            for i in range(n_hist-1):
                res = self.potential_history[-(i+1)] - self.potential_history[-(i+2)]
                residuals.append(res.flatten())
            
            residuals = np.array(residuals).T  # Shape: (n_sites, n_hist-1)
            
            # Current residual
            current_res = (phi_new - phi_old).flatten()
            
            # Solve linear system for Anderson coefficients
            if residuals.shape[1] > 0:
                # Solve: residuals @ c = current_res for coefficients c
                c, _, _, _ = np.linalg.lstsq(residuals, current_res, rcond=None)
                c = np.clip(c, -2.0, 2.0)  # Prevent wild extrapolation
                
                # Compute Anderson mixed potential
                phi_anderson = phi_new.copy()
                for i, coeff in enumerate(c):
                    if i < len(self.potential_history):
                        phi_anderson -= coeff * (self.potential_history[-(i+1)] - self.potential_history[-(i+2)])
                
                return phi_anderson
            else:
                return self.linear_mixing(phi_new, phi_old)
                
        except Exception:
            # Fall back to linear mixing if Anderson fails
            return self.linear_mixing(phi_new, phi_old)
    
    def compute_charge_density(self, 
                              fsys,
                              params: Dict[str, Any],
                              mu_left: float,
                              mu_right: float,
                              temperature: float = 0.0,
                              energy_span: float = 0.2,
                              n_energy_points: int = 21) -> np.ndarray:
        try:
            # Extract matrices from Kwant system - use average chemical potential as energy
            energy = (mu_left + mu_right) / 2
            
            # Debug: Check if potential is in params
            potential = params.get('electrostatic_potential', None)
            if potential is not None:
                print(f"Debug NEGF: Using potential with range [{np.min(potential):.3e}, {np.max(potential):.3e}]")
            else:
                print("Debug NEGF: No potential in params!")
            
            H_device, H_leads, V_couplings = extract_kwant_matrices(fsys, energy, leads_kwant_supported=None, params=params)
            
            # Create NEGF solver
            negf = NEGFSolver(H_device, H_leads, V_couplings)
            
            # Compute charge density under bias
            charge_density = negf.charge_density_at_bias(
                mu_left, mu_right, temperature, energy_span, n_energy_points
            )
            
            # Validate charge density
            if charge_density is None or len(charge_density) == 0:
                raise ValueError("NEGF returned empty charge density")
            
            # Ensure charge density matches expected size
            if len(charge_density) != self.n_sites:
                # Try to map or resize charge density to match lattice sites
                if len(charge_density) > self.n_sites:
                    # Truncate if too large
                    charge_density = charge_density[:self.n_sites]
                else:
                    # Pad if too small
                    padded = np.zeros(self.n_sites)
                    padded[:len(charge_density)] = charge_density
                    charge_density = padded
            
            return charge_density
            
        except Exception as e:
            warnings.warn(f"NEGF charge density calculation failed: {e}")
            # Fallback to uniform background charge
            return np.zeros(self.n_sites)
    
    def update_hamiltonian_potential(self, 
                                   fsys,
                                   potential: np.ndarray,
                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update system parameters with new electrostatic potential.
        The potential will be applied to graphene onsite energies via the 
        graphene_onsite_with_potential function.
        
        Parameters:
        -----------
        fsys : kwant.system.FiniteSystem
            Finalized Kwant system
        potential : np.ndarray
            Electrostatic potential at each site (in eV)
        params : dict
            Current system parameters
            
        Returns:
        --------
        dict
            Updated parameters with potential included
        """
        # Copy parameters
        new_params = params.copy()
        
        # Store potential for onsite functions to use
        new_params['electrostatic_potential'] = potential
        
        # Create site index map for the onsite functions to use
        # This maps each site to its index in the potential array
        if not hasattr(self, '_site_index_map'):
            # Handle different ways fsys might store sites
            if hasattr(fsys, 'sites'):
                sites_attr = getattr(fsys, 'sites')
                sites_list = list(sites_attr() if callable(sites_attr) else sites_attr)
            elif hasattr(fsys, 'graph'):
                sites_list = list(fsys.graph.nodes())
            else:
                raise AttributeError("Cannot access sites in finalized system")

            self._site_index_map = {site: i for i, site in enumerate(sites_list)}
        
        new_params['site_index_map'] = self._site_index_map
        
        return new_params
    
    def check_convergence(self, phi_new: np.ndarray, phi_old: np.ndarray) -> Tuple[bool, float]:
        """
        Check SCF convergence based on potential difference.
        
        Parameters:
        -----------
        phi_new, phi_old : np.ndarray
            New and old potentials
            
        Returns:
        --------
        Tuple[bool, float]
            (converged, max_difference)
        """
        if phi_old is None:
            return False, np.inf
            
        diff = np.abs(phi_new - phi_old)
        max_diff = np.max(diff)
        rms_diff = np.sqrt(np.mean(diff**2))
        
        # Store convergence history
        self.convergence_history.append({
            'iteration': self.iteration_count,
            'max_diff': max_diff,
            'rms_diff': rms_diff
        })
        
        converged = max_diff < self.tolerance
        return converged, max_diff
    
    def scf_loop(self,
                fsys,
                initial_params: Dict[str, Any],
                bias_voltage: float,
                temperature: float = 0.0,
                initial_potential: Optional[np.ndarray] = None,
                verbose: bool = True,
                compute_current_each_iter: bool = True,
                current_tolerance: Optional[float] = 1e-3,
                require_both_converged: bool = True,
                iteration_energy_points: int = 51,
                min_iterations: int = 3,
                post_validate: bool = True) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """
        Main SCF loop for self-consistent transport calculation.
        
        Parameters:
        -----------
        fsys : kwant.system.FiniteSystem
            Finalized Kwant system
        initial_params : dict
            Initial system parameters
        bias_voltage : float
            Applied bias voltage
        temperature : float
            Temperature in K
        initial_potential : np.ndarray, optional
            Initial guess for electrostatic potential
        verbose : bool
            Print convergence information
            
        Returns:
        --------
        Tuple containing:
        - converged_potential: Final electrostatic potential
        - converged_params: Final system parameters
        - converged: Whether SCF converged
        """
        # Initialize
        self.iteration_count = 0
        self.convergence_history = []
        self.potential_history = []
        self.charge_history = []
        self.current_history = []
        
        # Lead chemical potentials
        mu_left = bias_voltage / 2
        mu_right = -bias_voltage / 2
        
        # Initial potential guess
        if initial_potential is None:
            phi_old = np.zeros(self.n_sites)
        else:
            phi_old = np.asarray(initial_potential)
        
        current_params = initial_params.copy()
        converged = False
        
        if verbose:
            print("Starting SCF loop...")
            print(f"{'Iter':>4} {'Max Δφ':>12} {'RMS Δφ':>12} {'Status':>12}")
            print("-" * 50)
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            # Step 1: Update Hamiltonian with current potential
            current_params = self.update_hamiltonian_potential(fsys, phi_old, current_params)
            
            # Step 2: Compute charge density using NEGF
            try:
                if verbose:
                    print(f"  Iter {iteration+1}: computing charge density (NEGF)...")
                charge_density = self.compute_charge_density(
                    fsys, current_params, mu_left, mu_right, temperature
                )
                self.charge_history.append(charge_density.copy())
                
            except Exception as e:
                warnings.warn(f"SCF iteration {iteration+1} failed in charge calculation: {e}")
                # Continue to next iteration with zero charge to allow Poisson to run
                charge_density = np.zeros(self.n_sites)
            
            # Step 3: Solve Poisson equation for new potential
            try:
                if verbose:
                    print(f"  Iter {iteration+1}: solving Poisson...")
                phi_new = self.poisson_solver.solve_graphene(
                    charge_density, bias_voltage, (mu_left, mu_right)
                )
                
                # Validate potential result
                if phi_new is None:
                    raise ValueError("Poisson solver returned None")
                if isinstance(phi_new, tuple):
                    raise TypeError(f"Poisson solver returned tuple instead of array: {phi_new}")
                if len(phi_new) != len(charge_density):
                    raise ValueError(f"Potential size {len(phi_new)} != charge density size {len(charge_density)}")
                
                # Debug: Check if Poisson solver is returning meaningful results
                if verbose and iteration <= 2:  # Show first few iterations
                    print(f"Debug iter {iteration+1}: Charge density range: [{np.min(charge_density):.3e}, {np.max(charge_density):.3e}]")
                    print(f"Debug iter {iteration+1}: New potential range: [{np.min(phi_new):.3e}, {np.max(phi_new):.3e}]")
                    print(f"Debug iter {iteration+1}: Old potential range: [{np.min(phi_old):.3e}, {np.max(phi_old):.3e}]")
                    print(f"Debug iter {iteration+1}: Potential std dev: {np.std(phi_new):.3e}")
                    if iteration > 0:
                        charge_change = np.max(np.abs(charge_density - self.charge_history[-1]))
                        print(f"Debug iter {iteration+1}: Max charge change from prev: {charge_change:.3e}")
                    
            except Exception as e:
                warnings.warn(f"SCF iteration {iteration+1} failed in Poisson solve: {e}")
                break
            
            # Step 4: Compute current using the updated potential (per-iteration)
            current_value = None
            if compute_current_each_iter:
                try:
                    # Update params with phi_new for current evaluation
                    params_for_current = self.update_hamiltonian_potential(fsys, phi_new, current_params)
                    # Prefer C++ backend if available for I(V)
                    try:
                        H_device, H_leads, V_couplings = extract_kwant_matrices(fsys, (mu_left+mu_right)/2, params_for_current)
                        negf = NEGFSolver(H_device, H_leads, V_couplings)
                        I_cpp = negf.finite_bias_current_cpp(mu_left, mu_right, temperature, (mu_left+mu_right)/2,
                                                             E_span=0.2, NE=max(21, iteration_energy_points))
                        if I_cpp is not None:
                            current_value = I_cpp
                        else:
                            raise RuntimeError("C++ current unavailable")
                    except Exception:
                        # Fallback to Python finite-bias calculation
                        current_value, _ = compute_finite_bias_current(
                            fsys,
                            params_for_current,
                            bias_voltage,
                            temperature=temperature,
                            n_energy_points=iteration_energy_points,
                            verbose=False
                        )
                    self.current_history.append(current_value)
                except Exception as e:
                    warnings.warn(f"SCF iteration {iteration+1}: current evaluation failed: {e}")
                    # Append NaN to indicate failure but continue
                    self.current_history.append(np.nan)

            # Step 5: Check convergence (potential and optionally current)
            try:
                converged, max_diff = self.check_convergence(phi_new, phi_old)
            except Exception as e:
                warnings.warn(f"SCF iteration {iteration+1} failed in convergence check: {e}")
                break
            
            if verbose:
                rms_diff = np.sqrt(np.mean((phi_new - phi_old)**2)) if phi_old is not None else np.inf
                # Evaluate current convergence
                current_ok = True
                rel_change = None
                if compute_current_each_iter and current_tolerance is not None and len(self.current_history) >= 2:
                    c_new = self.current_history[-1]
                    c_old = self.current_history[-2]
                    if np.isfinite(c_new) and np.isfinite(c_old):
                        denom = max(abs(c_new), 1e-12)
                        rel_change = abs(c_new - c_old) / denom
                        current_ok = rel_change < current_tolerance
                # Decide overall convergence
                overall_converged = converged and (current_ok if require_both_converged else True) and ((iteration+1) >= min_iterations)
                status = "CONVERGED  " if overall_converged else "RUNNING  "
                extra = f"  I={current_value:.3e} A" if current_value is not None else ""
                if rel_change is not None:
                    extra += f"  ΔI/I={rel_change:.3e}"
                print(f"{iteration+1:4d} {max_diff:12.6e} {rms_diff:12.6e} {status:>12}{extra}")
            else:
                # Non-verbose: still determine overall convergence
                current_ok = True
                if compute_current_each_iter and current_tolerance is not None and len(self.current_history) >= 2:
                    c_new = self.current_history[-1]
                    c_old = self.current_history[-2]
                    if np.isfinite(c_new) and np.isfinite(c_old):
                        denom = max(abs(c_new), 1e-12)
                        current_ok = abs(c_new - c_old) / denom < current_tolerance
                overall_converged = converged and (current_ok if require_both_converged else True) and ((iteration+1) >= min_iterations)
            
            if overall_converged:
                # Optional post-validation: one extra fixed-point check using phi_new
                if post_validate:
                    try:
                        # Recompute with phi_new as the starting potential (no mixing)
                        params_val = self.update_hamiltonian_potential(fsys, phi_new, current_params)
                        charge_val = self.compute_charge_density(fsys, params_val, mu_left, mu_right, temperature)
                        phi_val = self.poisson_solver.solve_graphene(charge_val, bias_voltage, (mu_left, mu_right))
                        # Compute current with validated params
                        current_val, _ = (compute_finite_bias_current(
                            fsys, params_val, bias_voltage, temperature=temperature,
                            n_energy_points=iteration_energy_points, verbose=False
                        ) if compute_current_each_iter else (np.nan, np.nan))
                        # Check deltas
                        phi_delta = np.max(np.abs(phi_val - phi_new)) if phi_val is not None else np.inf
                        current_delta_ok = True
                        if compute_current_each_iter and np.isfinite(current_val) and current_value is not None and np.isfinite(current_value):
                            denom = max(abs(current_val), 1e-12)
                            current_delta_ok = abs(current_val - current_value) / denom < current_tolerance
                        if verbose:
                            print(f"  Validation: max|Δφ|={phi_delta:.3e}" + (f"  ΔI/I={abs(current_val-current_value)/max(abs(current_val),1e-12):.3e}" if compute_current_each_iter and np.isfinite(current_val) and current_value is not None and np.isfinite(current_value) else ""))
                        if (phi_delta < self.tolerance) and (current_delta_ok or not require_both_converged):
                            if verbose:
                                print(f"SCF converged in {iteration+1} iterations (validated).")
                            break
                        else:
                            # Not a fixed point yet, continue iterating
                            overall_converged = False
                            if verbose:
                                print("  Validation failed -> continue iterations")
                    except Exception as e:
                        warnings.warn(f"Post-validation failed: {e}, continuing iterations")
                        overall_converged = False
                if overall_converged:
                    break
            
            # Step 6: Mix potentials for next iteration (update Hamiltonian for next loop)
            if iteration > 0:
                # Use Anderson mixing after first few iterations for better convergence
                if iteration > 3 and len(self.potential_history) >= 2:
                    phi_mixed = self.anderson_mixing(phi_new, phi_old)
                else:
                    phi_mixed = self.linear_mixing(phi_new, phi_old)
            else:
                phi_mixed = phi_new
            
            self.potential_history.append(phi_mixed.copy())
            phi_old = phi_mixed
        
        else:
            if verbose:
                print(f"SCF did not converge after {self.max_iterations} iterations.")
            converged = False
        
        # Final parameters with converged potential
        final_params = self.update_hamiltonian_potential(fsys, phi_old, current_params)
        
        return phi_old, final_params, converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get detailed convergence information.
        
        Returns:
        --------
        dict
            Convergence statistics and history
        """
        if not self.convergence_history:
            return {}
        
        return {
            'iterations': self.iteration_count,
            'converged': len(self.convergence_history) > 0 and 
                        self.convergence_history[-1]['max_diff'] < self.tolerance,
            'final_max_diff': self.convergence_history[-1]['max_diff'],
            'final_rms_diff': self.convergence_history[-1]['rms_diff'],
            'history': self.convergence_history,
            'potential_history': self.potential_history,
            'charge_history': self.charge_history,
            'current_history': getattr(self, 'current_history', [])
        }

# Convenience functions for easy integration

def fermi_function(energy: float, mu: float, temperature: float) -> float:
    """
    Fermi-Dirac distribution function.
    
    Parameters:
    -----------
    energy : float
        Energy in eV
    mu : float
        Chemical potential in eV
    temperature : float
        Temperature in K
        
    Returns:
    --------
    float
        Fermi function f(E) = 1 / (1 + exp((E-μ)/(kB*T)))
    """
    if temperature < 1e-12:  # T → 0 limit
        return 1.0 if energy <= mu else 0.0
    
    kB_eV = 8.617333262e-5  # Boltzmann constant in eV/K
    x = (energy - mu) / (kB_eV * temperature)
    
    # Handle numerical overflow/underflow
    if x > 50:
        return 0.0
    elif x < -50:
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(x))


def compute_finite_bias_current(fsys, 
                               params: Dict[str, Any],
                               bias_voltage: float,
                               temperature: float = 300.0,
                               energy_window: float = 0.5,
                               n_energy_points: int = 101,
                               verbose: bool = False) -> Tuple[float, float]:
    from negf_core import extract_kwant_matrices, NEGFSolver
    
    # Lead chemical potentials
    mu_left = bias_voltage / 2
    mu_right = -bias_voltage / 2
    
    # Get Fermi energy from parameters
    E_fermi = params.get('EF', params.get('E', 0.0))
    
    # For finite bias, focus energy window around the bias range
    bias_span = abs(mu_left - mu_right)  # Total bias window
    if bias_span > 0:
        # Energy window should span the bias plus some thermal broadening
        kB_T = 8.617333262e-5 * max(temperature, 10.0)  # At least 10K for numerical stability
        energy_window = max(energy_window, 3*bias_span + 6*kB_T)  # 3x bias + 6kT thermal tails
        
        # Center around the average of the chemical potentials
        E_center = (mu_left + mu_right) / 2
        E_min = E_center - energy_window / 2
        E_max = E_center + energy_window / 2
    else:
        # No bias case - use original behavior
        E_min = E_fermi - energy_window / 2
        E_max = E_fermi + energy_window / 2
    
    energies = np.linspace(E_min, E_max, n_energy_points)
    dE = energies[1] - energies[0] if len(energies) > 1 else 0.0
    
    if verbose:
        print(f"  Finite bias integration: {n_energy_points} points from {E_min:.4f} to {E_max:.4f} eV")
        print(f"  μ_L = {mu_left:.6f} eV, μ_R = {mu_right:.6f} eV, T = {temperature:.1f} K")
        print(f"  Bias window: [{min(mu_left, mu_right):.6f}, {max(mu_left, mu_right):.6f}] eV")
    
    current_integrand = []
    transmissions = []
    
    try:
        for i, energy in enumerate(energies):
            # Extract matrices at this energy
            H_device, H_leads, V_couplings = extract_kwant_matrices(
                fsys, energy, leads_kwant_supported=None, params=params
            )
            
            # Create NEGF solver
            negf = NEGFSolver(H_device, H_leads, V_couplings)
            
            # Compute transmission at this energy
            T_E = negf.transmission(energy, lead_i=0, lead_j=1)
            transmissions.append(T_E)
            
            # Fermi functions for left and right leads
            f_L = fermi_function(energy, mu_left, temperature)
            f_R = fermi_function(energy, mu_right, temperature)
            
            # Current integrand: T(E) * [f_L(E) - f_R(E)]
            integrand = T_E * (f_L - f_R)
            current_integrand.append(integrand)
            
            if verbose and (i == 0 or i == len(energies)-1 or i % (len(energies)//10) == 0 or 
                           (min(mu_left, mu_right) <= energy <= max(mu_left, mu_right))):
                print(f"    E = {energy:.4f} eV: T = {T_E:.6f}, f_L = {f_L:.6f}, f_R = {f_R:.6f}, integrand = {integrand:.6e}")
    
    except Exception as e:
        if verbose:
            print(f"  Warning: Energy integration failed at E = {energy:.4f} eV: {e}")
        # Fallback to linear response
        try:
            H_device, H_leads, V_couplings = extract_kwant_matrices(
                fsys, E_fermi, leads_kwant_supported=None, params=params
            )
            negf = NEGFSolver(H_device, H_leads, V_couplings)
            T_fermi = negf.transmission(E_fermi, lead_i=0, lead_j=1)
            
            # Linear response current: G * V where G = (2e²/h) * T
            e_charge = 1.602176634e-19
            h_planck = 6.62607015e-34
            conductance = T_fermi * 2 * e_charge**2 / h_planck
            current = conductance * bias_voltage
            
            return current, T_fermi
            
        except Exception as fallback_error:
            if verbose:
                print(f"  Fallback also failed: {fallback_error}")
            return 0.0, 0.0
    
    # Integrate using trapezoidal rule
    if len(current_integrand) > 1:
        integral = np.trapz(current_integrand, dx=dE)
    else:
        integral = current_integrand[0] * dE if current_integrand else 0.0
    
    # Convert to current: I = (2e/h) * integral
    e_charge = 1.602176634e-19  # Elementary charge in C
    h_planck = 6.62607015e-34   # Planck constant in J⋅s
    prefactor = e_charge**2 / h_planck
    
    current = prefactor * integral  # Current in Amperes
    avg_transmission = np.mean(transmissions) if transmissions else 0.0
    
    if verbose:
        print(f"  Integrated current: I = {current:.6e} A")
        print(f"  Average transmission: <T> = {avg_transmission:.6f}")
        print(f"  Effective conductance: G = I/V = {current/bias_voltage if abs(bias_voltage) > 1e-12 else 0:.6e} S")
        
        # Current magnitude analysis
        print(f"  WARNING: **PHYSICS CHECK**: Current magnitude = {current:.2e} A")
        if abs(current) > 1e9:
            print(f"      This suggests current PER UNIT LENGTH (very large for single device)")
            print(f"      For realistic device lengths:")
            print(f"      • 1 nm device: {current*1e-9:.2e} A = {current*1e-9*1e12:.1f} pA")
            print(f"      • 10 nm device: {current*1e-8:.2e} A = {current*1e-8*1e9:.1f} nA")
        elif 1e-12 <= abs(current) <= 1e-6:
            print(f"      VALID: Magnitude reasonable for molecular/nanoelectronics device")
        elif abs(current) < 1e-12:
            print(f"      Current very small - may need higher bias or temperature")
        else:
            print(f"      Current quite large - check system parameters and units")
    
    return current, avg_transmission


def scf_conductance(fsys, 
                   lattice_sites: np.ndarray,
                   params: Dict[str, Any],
                   bias_voltage: float,
                   temperature: float = 0.0,
                   scf_tolerance: float = 1e-6,
                   max_scf_iterations: int = 50,
                   use_finite_bias: bool = True,
                   verbose: bool = False,
                   current_tolerance: Optional[float] = 1e-3,
                   require_both_converged: bool = True) -> Tuple[float, float, bool]:
    # Initialize SCF solver
    scf = SCFSolver(
        lattice_sites=lattice_sites,
        max_iterations=max_scf_iterations,
        tolerance=scf_tolerance
    )
    
    # Run SCF loop
    try:
        phi_scf, params_scf, converged = scf.scf_loop(
            fsys, params, bias_voltage, temperature,
            verbose=verbose,
            compute_current_each_iter=use_finite_bias,
            current_tolerance=current_tolerance,
            require_both_converged=require_both_converged,
            iteration_energy_points=51
        )
        
        if not converged:
            warnings.warn("SCF did not converge, results may be inaccurate")
        
        if use_finite_bias:
            # Compute final current using finite bias integration
            current, avg_transmission = compute_finite_bias_current(
                fsys, params_scf, bias_voltage, temperature, verbose=verbose
            )
            
            # Convert to conductance for compatibility (dI/dV at small bias)
            if abs(bias_voltage) > 1e-12:
                conductance = current / bias_voltage  # I/V for finite bias
            else:
                # Fallback to linear response for zero bias
                from negf_core import extract_kwant_matrices, NEGFSolver
                energy = params.get('EF', params.get('E', 0.0))
                H_device, H_leads, V_couplings = extract_kwant_matrices(
                    fsys, energy, leads_kwant_supported=None, params=params_scf
                )
                negf = NEGFSolver(H_device, H_leads, V_couplings)
                avg_transmission = negf.transmission(energy, lead_i=0, lead_j=1)
                conductance = avg_transmission * 2 * 1.602176634e-19**2 / 6.62607015e-34  # e²/h in SI
        else:
            # Use linear response (original behavior)
            from negf_core import extract_kwant_matrices, NEGFSolver
            energy = params.get('EF', params.get('E', 0.0))
            H_device, H_leads, V_couplings = extract_kwant_matrices(
                fsys, energy, leads_kwant_supported=None, params=params_scf
            )
            negf = NEGFSolver(H_device, H_leads, V_couplings)
            avg_transmission = negf.transmission(energy, lead_i=0, lead_j=1)
            conductance = avg_transmission * 2 * 1.602176634e-19**2 / 6.62607015e-34  # e²/h in SI
        
        return conductance, avg_transmission, converged
        
    except Exception as e:
        warnings.warn(f"SCF conductance calculation failed: {e}")
        return 0.0, 0.0, False