"""
Clean electronic structure analysis functions for graphene SW defect systems
These replace the overly complex functions in the main script
"""

import numpy as np
import matplotlib.pyplot as plt
import kwant

def plot_clean_transmission(fsys, args, ax):
    """Simple, clean transmission plot"""
    try:
        # Parameters
        params = dict(t=args.t, eps_sw=getattr(args, 'eps_sw', 0), 
                     alpha1_sw=getattr(args, 'alpha1_sw', 0.01),
                     alpha2_sw=getattr(args, 'alpha2_sw', -0.05),
                     X=0.0, g=args.g, Bx=args.Bx, By=args.By, Bz=args.Bz,
                     Vimp_eta2=getattr(args, 'Vimp_eta2', 0.5), 
                     Vimp_side=getattr(args, 'Vimp_side', 0.2))
        params['electrostatic_potential'] = None
        params['site_index_map'] = None
        
        # Energy range
        energies = np.linspace(args.EF - 0.1, args.EF + 0.1, 101)
        
        # Calculate transmission
        transmissions = []
        for E in energies:
            try:
                sm = kwant.smatrix(fsys, energy=E, params=params)
                T = sm.transmission(0, 1)
                transmissions.append(T)
            except:
                transmissions.append(0.0)
        
        # Plot
        ax.plot(energies, transmissions, 'b-', linewidth=2)
        ax.fill_between(energies, 0, transmissions, alpha=0.3, color='blue')
        
        # Add Fermi level
        ax.axvline(args.EF, color='red', linestyle='--', linewidth=2, label='Fermi Level')
        
        # Bias window
        V = getattr(args, 'bias_voltage', getattr(args, 'Vb', 0.001))
        ax.axvspan(args.EF - V/2, args.EF + V/2, alpha=0.2, color='orange', 
                  label=f'Bias ({V*1000:.1f} mV)')
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Transmission')
        ax.set_title(f'Transmission Spectrum\n{args.lanthanide} + SW Defect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Transmission failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes)

def plot_dos_histogram(fsys_sw, fsys_pr, args, ax):
    """Simple DOS comparison using eigenvalue histograms"""
    try:
        # Parameters
        params = dict(t=args.t, eps_sw=getattr(args, 'eps_sw', 0), 
                     alpha1_sw=getattr(args, 'alpha1_sw', 0.01),
                     alpha2_sw=getattr(args, 'alpha2_sw', -0.05),
                     X=0.0, g=args.g, Bx=args.Bx, By=args.By, Bz=args.Bz,
                     Vimp_eta2=getattr(args, 'Vimp_eta2', 0.5), 
                     Vimp_side=getattr(args, 'Vimp_side', 0.2))
        params['electrostatic_potential'] = None
        params['site_index_map'] = None
        
        # Get eigenvalues
        ham_sw = fsys_sw.hamiltonian_submatrix(sparse=False, params=params)
        eigs_sw = np.real(np.linalg.eigvals(ham_sw))
        
        # Filter to reasonable range
        E_range = (args.EF - 0.2, args.EF + 0.2)
        eigs_sw = eigs_sw[(eigs_sw >= E_range[0]) & (eigs_sw <= E_range[1])]
        
        # Plot histogram
        bins = np.linspace(E_range[0], E_range[1], 41)
        ax.hist(eigs_sw, bins=bins, alpha=0.7, color='red', 
               label=f'SW + {args.lanthanide} ({len(eigs_sw)} states)', density=True)
        
        # Pristine system if available
        if fsys_pr:
            ham_pr = fsys_pr.hamiltonian_submatrix(sparse=False, params=params)
            eigs_pr = np.real(np.linalg.eigvals(ham_pr))
            eigs_pr = eigs_pr[(eigs_pr >= E_range[0]) & (eigs_pr <= E_range[1])]
            
            ax.hist(eigs_pr, bins=bins, alpha=0.5, color='blue', 
                   label=f'Pristine + {args.lanthanide} ({len(eigs_pr)} states)', 
                   density=True, histtype='step', linewidth=2)
        
        # Add Fermi level
        ax.axvline(args.EF, color='black', linestyle='--', linewidth=2, label='Fermi Level')
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Density of States')
        ax.set_title(f'DOS Comparison\n{args.lanthanide} System')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'DOS failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes)

def plot_wavefunction_simple(fsys, args, ax):
    """Simple wavefunction plot near Fermi level"""
    try:
        # Parameters
        params = dict(t=args.t, eps_sw=getattr(args, 'eps_sw', 0), 
                     alpha1_sw=getattr(args, 'alpha1_sw', 0.01),
                     alpha2_sw=getattr(args, 'alpha2_sw', -0.05),
                     X=0.0, g=args.g, Bx=args.Bx, By=args.By, Bz=args.Bz,
                     Vimp_eta2=getattr(args, 'Vimp_eta2', 0.5), 
                     Vimp_side=getattr(args, 'Vimp_side', 0.2))
        params['electrostatic_potential'] = None
        params['site_index_map'] = None
        
        # Get Hamiltonian and solve
        ham = fsys.hamiltonian_submatrix(sparse=False, params=params)
        eigenvals, eigenvecs = np.linalg.eigh(ham)
        
        # Find state closest to Fermi level
        fermi_idx = np.argmin(np.abs(eigenvals - args.EF))
        state = eigenvecs[:, fermi_idx]
        energy = eigenvals[fermi_idx]
        
        # Get positions
        sites = list(fsys.sites)
        positions = np.array([fsys.pos(site) for site in sites])
        
        # Calculate probability density
        prob = np.abs(state)**2
        
        # Plot
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=prob, s=60, cmap='hot', alpha=0.8)
        plt.colorbar(scatter, ax=ax, label='|ψ|²')
        
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_title(f'Wavefunction Near Fermi Level\nE = {energy:.3f} eV')
        ax.set_aspect('equal')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Wavefunction failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes)

def plot_iv_curve(fsys, args, ax):
    """Simple I-V characteristic"""
    try:
        # Bias voltages
        voltages = np.linspace(-0.01, 0.01, 21)  # ±10 mV
        currents = []
        
        # Parameters
        base_params = dict(t=args.t, eps_sw=getattr(args, 'eps_sw', 0), 
                          alpha1_sw=getattr(args, 'alpha1_sw', 0.01),
                          alpha2_sw=getattr(args, 'alpha2_sw', -0.05),
                          X=0.0, g=args.g, Bx=args.Bx, By=args.By, Bz=args.Bz,
                          Vimp_eta2=getattr(args, 'Vimp_eta2', 0.5), 
                          Vimp_side=getattr(args, 'Vimp_side', 0.2))
        
        for V in voltages:
            try:
                params = base_params.copy()
                params['electrostatic_potential'] = None
                params['site_index_map'] = None
                
                # Simple approximation: current from transmission at Fermi level
                sm = kwant.smatrix(fsys, energy=args.EF, params=params)
                T = sm.transmission(0, 1)
                
                # I ≈ (2e²/h) * T * V  for small bias
                I = 2 * (1.602e-19)**2 / 6.626e-34 * T * V
                currents.append(I * 1e6)  # Convert to μA
            except:
                currents.append(0.0)
        
        # Plot
        ax.plot(voltages * 1000, currents, 'g-o', linewidth=2, markersize=4)
        
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel('Current (μA)')
        ax.set_title(f'I-V Characteristic\n{args.lanthanide} + SW Defect')
        ax.grid(True, alpha=0.3)
        
        # Calculate conductance
        if len(currents) > 1:
            G = np.polyfit(voltages, np.array(currents)/1e6, 1)[0]  # dI/dV in A/V
            ax.text(0.02, 0.98, f'Conductance ≈ {G*1e6:.2f} μS', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
    except Exception as e:
        ax.text(0.5, 0.5, f'I-V failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes)