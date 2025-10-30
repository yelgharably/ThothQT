"""
Graphene Quantum Sensor Example
================================
Demonstrate using KWANT→ThoothQT bridge for quantum sensing applications.

This example shows how to:
1. Build graphene nanoribbons with KWANT
2. Add physical impurities (vacancies, dopants, adsorbates)
3. Apply electric fields
4. Calculate transmission and analyze sensitivity
"""

import numpy as np
import kwant
import matplotlib.pyplot as plt
from kwant_to_toothqt import kwant_to_toothqt, add_onsite_potential, add_positional_field
from toothqt_production import NEGFEngine


def build_graphene_ribbon(width: int, length: int, a: float = 1.42, t: float = -2.7):
    """Build graphene zigzag nanoribbon using KWANT."""
    graphene = kwant.lattice.honeycomb(a, name='graphene')
    a_sub, b_sub = graphene.sublattices
    
    # Device
    sys = kwant.Builder()
    for ix in range(length):
        for iy in range(width):
            sys[a_sub(ix, iy)] = 0
            sys[b_sub(ix, iy)] = 0
    sys[graphene.neighbors()] = t
    
    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((1, 0))))
    for iy in range(width):
        lead[a_sub(0, iy)] = 0
        lead[b_sub(0, iy)] = 0
    lead[graphene.neighbors()] = t
    
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
    
    # Get positions before finalizing
    sites = list(sys.sites())
    positions = np.array([site.pos for site in sites])
    
    return sys.finalized(), positions


def find_central_sites(positions: np.ndarray, radius: float) -> list:
    """Find sites within radius of device center."""
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    return np.where(distances < radius)[0].tolist()


def plot_transmission_spectrum(devices: dict, E_range: np.ndarray, 
                               labels: dict, title: str = "Transmission Spectrum"):
    """Plot transmission vs energy for multiple devices."""
    plt.figure(figsize=(10, 6))
    
    for name, device in devices.items():
        negf = NEGFEngine(device, Temp=300.0)
        T_list = []
        
        print(f"Computing {name}...", end=" ")
        for E in E_range:
            T = negf.transmission(E)
            T_list.append(T)
        print(f"Done (T_max = {max(T_list):.3f})")
        
        plt.plot(E_range, T_list, '-', linewidth=2, label=labels[name])
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Transmission (G₀)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


# ==============================================================================
# EXAMPLE 1: Pristine Graphene Ribbon
# ==============================================================================
print("=" * 80)
print("EXAMPLE 1: Pristine Graphene Zigzag Nanoribbon")
print("=" * 80)

width = 4
length = 20

print(f"\nBuilding graphene ribbon (width={width}, length={length})...")
fsys, positions = build_graphene_ribbon(width, length)
print(f"  Total sites: {len(positions)}")

print("\nConverting to ThoothQT...")
device_clean = kwant_to_toothqt(fsys, Ef=0.0)

print("\nComputing transmission spectrum...")
E_range = np.linspace(-1.0, 1.0, 41)
devices_ex1 = {'clean': device_clean}
labels_ex1 = {'clean': 'Pristine graphene'}
fig1 = plot_transmission_spectrum(devices_ex1, E_range, labels_ex1, 
                                   "Pristine Graphene Zigzag Ribbon")
fig1.savefig('graphene_pristine.png', dpi=150)
print("Saved: graphene_pristine.png")


# ==============================================================================
# EXAMPLE 2: Point Defect (Impurity/Dopant)
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Graphene with Point Defect")
print("=" * 80)

# Add single impurity at center
center_sites = find_central_sites(positions, radius=1.0)
print(f"\nAdding impurity at {len(center_sites)} central sites")

impurity_strengths = [0.0, 0.5, 1.0, 2.0]  # eV
devices_ex2 = {}
labels_ex2 = {}

for V in impurity_strengths:
    if V == 0.0:
        devices_ex2[f'V{V}'] = device_clean
        labels_ex2[f'V{V}'] = 'Clean'
    else:
        device_imp = add_onsite_potential(device_clean, center_sites, V)
        devices_ex2[f'V{V}'] = device_imp
        labels_ex2[f'V{V}'] = f'Impurity V={V} eV'

print("\nComputing transmission with varying impurity strength...")
E_range_imp = np.linspace(-0.5, 0.5, 21)
fig2 = plot_transmission_spectrum(devices_ex2, E_range_imp, labels_ex2,
                                   "Effect of Point Defect on Transmission")
fig2.savefig('graphene_impurity.png', dpi=150)
print("Saved: graphene_impurity.png")


# ==============================================================================
# EXAMPLE 3: Electric Field (Linear Potential)
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Graphene under Electric Field")
print("=" * 80)

field_strengths = [0.0, 0.01, 0.02, 0.05]  # eV/Å
devices_ex3 = {}
labels_ex3 = {}

for F in field_strengths:
    if F == 0.0:
        devices_ex3[f'F{F}'] = device_clean
        labels_ex3[f'F{F}'] = 'No field'
    else:
        # Field in x-direction (transport direction)
        field_vector = np.array([F, 0])
        device_field = add_positional_field(device_clean, positions, field_vector)
        devices_ex3[f'F{F}'] = device_field
        labels_ex3[f'F{F}'] = f'Field = {F} eV/Å'
        print(f"  Applied field: {F} eV/Å in x-direction")

print("\nComputing transmission with varying field strength...")
fig3 = plot_transmission_spectrum(devices_ex3, E_range_imp, labels_ex3,
                                   "Effect of Electric Field on Transmission")
fig3.savefig('graphene_field.png', dpi=150)
print("Saved: graphene_field.png")


# ==============================================================================
# EXAMPLE 4: Quantum Sensing - Detect Adsorbate
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Quantum Sensing Application")
print("=" * 80)
print("Simulating molecular adsorbate detection via transmission change")

# Different adsorbate types (different onsite shifts)
adsorbates = {
    'H2O': 0.2,   # Water molecule (weak)
    'NH3': 0.5,   # Ammonia (moderate)
    'NO2': 1.0,   # Nitrogen dioxide (strong electron acceptor)
}

E_sense = 0.3  # Sensing energy (eV)

print(f"\nSensing energy: {E_sense} eV")
print("=" * 80)

# Baseline (clean)
negf_clean = NEGFEngine(device_clean, Temp=300.0)
T_baseline = negf_clean.transmission(E_sense)
print(f"Baseline transmission: {T_baseline:.6f}")

# Test each adsorbate
print("\nAdsorbate detection:")
print("-" * 80)
sensitivities = {}

for molecule, V_ads in adsorbates.items():
    # Add adsorbate (affects central sites)
    device_ads = add_onsite_potential(device_clean, center_sites, V_ads)
    
    # Measure transmission
    negf_ads = NEGFEngine(device_ads, Temp=300.0)
    T_ads = negf_ads.transmission(E_sense)
    
    # Calculate sensitivity
    delta_T = T_ads - T_baseline
    sensitivity = abs(delta_T / T_baseline) * 100 if T_baseline > 1e-10 else 0
    sensitivities[molecule] = sensitivity
    
    print(f"{molecule:>5s}: T = {T_ads:.6f}, ΔT = {delta_T:+.6f}, "
          f"Sensitivity = {sensitivity:.1f}%")

print("=" * 80)
print(f"\nBest sensitivity: {max(sensitivities, key=sensitivities.get)} "
      f"({max(sensitivities.values()):.1f}%)")


# ==============================================================================
# EXAMPLE 5: Multi-Defect System
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 5: Multiple Defects (Realistic Scenario)")
print("=" * 80)

# Create defects at different locations
n_sites = len(positions)
defect_sites = [
    n_sites // 4,      # Left defect
    n_sites // 2,      # Center defect
    3 * n_sites // 4   # Right defect
]
print(f"\nAdding defects at sites: {defect_sites}")

devices_ex5 = {'clean': device_clean}
labels_ex5 = {'clean': 'Clean'}

# Add defects incrementally
for i, site in enumerate(defect_sites, 1):
    device_def = add_onsite_potential(device_clean, defect_sites[:i], 0.8)
    devices_ex5[f'defect{i}'] = device_def
    labels_ex5[f'defect{i}'] = f'{i} defect{"s" if i > 1 else ""}'

print("\nComputing transmission with multiple defects...")
fig5 = plot_transmission_spectrum(devices_ex5, E_range_imp, labels_ex5,
                                   "Effect of Multiple Defects")
fig5.savefig('graphene_multi_defects.png', dpi=150)
print("Saved: graphene_multi_defects.png")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: KWANT→ThoothQT Bridge for Quantum Sensing")
print("=" * 80)
print("""
✓ Successfully built graphene nanoribbons with KWANT
✓ Converted to ThoothQT for fast NEGF calculations
✓ Added point defects (impurities, dopants)
✓ Applied electric fields (linear potentials)
✓ Simulated molecular adsorbate detection
✓ Analyzed multi-defect systems

Key Advantages:
  • KWANT: Flexible geometry, easy lattice construction
  • ThoothQT: Fast NEGF engine (2-5× faster than KWANT)
  • Easy modification: Add impurities/fields after conversion
  • Perfect for: Quantum sensing, defect physics, field effects

Next Steps:
  1. Modify Hamiltonians for your specific sensing application
  2. Sweep parameters (defect strength, position, field)
  3. Calculate I-V curves using NEGF current integration
  4. Optimize sensor geometry for maximum sensitivity
""")
print("=" * 80)
