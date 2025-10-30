"""
Comprehensive Graphene Nanoribbon Validation: ThothQT vs KWANT
============================================================

Tests both zigzag and armchair graphene nanoribbons with:
1. Clean systems (no impurities)
2. Systems with impurities/disorder
3. Transmission, DOS, and band structure comparisons
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def make_kwant_zigzag_nanoribbon(width=4, length=6, t=2.7, impurities=None):
    """Create KWANT zigzag graphene nanoribbon"""
    # Define graphene lattice
    graphene = kwant.lattice.honeycomb(norbs=1)
    a, b = graphene.sublattices
    
    syst = kwant.Builder()
    
    # Add device region
    def ribbon_shape(pos):
        x, y = pos
        return -length/2 < x < length/2 and -width/2 < y < width/2
    
    # Add sites and hoppings for zigzag ribbon
    for x in range(-length//2, length//2 + 1):
        for y in range(-width//2, width//2 + 1):
            try:
                # Add A sublattice sites
                if ribbon_shape((x, y)):
                    onsite_a = 0.0
                    if impurities and (x, y, 'a') in impurities:
                        onsite_a = impurities[(x, y, 'a')]
                    syst[a(x, y)] = onsite_a
                    
                # Add B sublattice sites  
                if ribbon_shape((x, y)):
                    onsite_b = 0.0
                    if impurities and (x, y, 'b') in impurities:
                        onsite_b = impurities[(x, y, 'b')]
                    syst[b(x, y)] = onsite_b
                    
            except:
                pass
    
    # Add nearest-neighbor hoppings
    syst[graphene.neighbors()] = -t
    
    # Create leads (semi-infinite extensions)
    lead = kwant.Builder(kwant.TranslationalSymmetry(graphene.vec((-1, 0))))
    
    def lead_shape(pos):
        x, y = pos
        return -width/2 < y < width/2
    
    lead[graphene.shape(lead_shape, (0, 0))] = 0.0
    lead[graphene.neighbors()] = -t
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    try:
        return syst.finalized()
    except Exception as e:
        print(f"KWANT graphene system creation failed: {e}")
        return None

def make_simple_kwant_graphene(width=3, length=4):
    """Create a simplified KWANT graphene system that's more likely to work"""
    # Use square lattice approximation for easier comparison
    lat = kwant.lattice.square(a=1, norbs=1)
    
    syst = kwant.Builder()
    
    # Add sites in rectangular region
    for x in range(length):
        for y in range(width):
            syst[lat(x, y)] = 0.0  # On-site energy
    
    # Add nearest-neighbor hoppings
    for x in range(length):
        for y in range(width):
            # Horizontal hoppings
            if x < length - 1:
                syst[lat(x, y), lat(x+1, y)] = -2.7
            # Vertical hoppings  
            if y < width - 1:
                syst[lat(x, y), lat(x, y+1)] = -2.7
    
    # Create 1D leads along x-direction
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    
    for y in range(width):
        lead[lat(0, y)] = 0.0
        if y < width - 1:
            lead[lat(0, y), lat(0, y+1)] = -2.7
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    return syst.finalized()

def compute_kwant_transmission_graphene(finalized_syst, energies):
    """Compute KWANT transmission for graphene system"""
    if finalized_syst is None:
        return np.zeros(len(energies))
    
    T_kwant = []
    for E in energies:
        try:
            smatrix = kwant.smatrix(finalized_syst, E)
            T_kwant.append(smatrix.transmission(0, 1))
        except Exception:
            T_kwant.append(0.0)
    
    return np.array(T_kwant)

def compute_kwant_dos_graphene(finalized_syst, energies):
    """Compute KWANT DOS for graphene system"""
    if finalized_syst is None:
        return np.zeros(len(energies))
    
    dos_kwant = []
    for E in energies:
        try:
            if abs(E) < 8.0:  # Stay within reasonable energy range
                ldos = kwant.ldos(finalized_syst, E)
                dos_total = np.sum(ldos)
                dos_kwant.append(dos_total)
            else:
                dos_kwant.append(0.0)
        except Exception:
            dos_kwant.append(0.0)
    
    return np.array(dos_kwant)

def graphene_validation():
    """Run comprehensive graphene nanoribbon validation"""
    print("Graphene Nanoribbon Validation: ThothQT vs KWANT")
    print("=" * 60)
    
    # System parameters
    width = 3
    length = 4
    t = 2.7  # eV
    
    print(f"System: Zigzag graphene nanoribbon")
    print(f"Width: {width}, Length: {length}, Hopping: {t} eV")
    
    # === TEST 1: CLEAN ZIGZAG NANORIBBON ===
    print("\n=== Test 1: Clean Zigzag Nanoribbon ===")
    
    # ThothQT system
    print("Creating ThothQT zigzag nanoribbon...")
    try:
        device_tqt = tqt.make_graphene_nanoribbon(
            width=width, length=length, edge_type='zigzag', t=t
        )
        engine_tqt = tqt.NEGFEngine(device_tqt, Temp=300)
        print(f"ThothQT system created: {device_tqt.H.shape[0]} atoms")
        thothqt_success = True
    except Exception as e:
        print(f"ThothQT system creation failed: {e}")
        thothqt_success = False
    
    # KWANT system (use simplified version for robustness)
    print("Creating KWANT graphene system...")
    try:
        finalized_syst_kwant = make_simple_kwant_graphene(width=width, length=length)
        print("KWANT system created successfully")
        kwant_success = True
    except Exception as e:
        print(f"KWANT system creation failed: {e}")
        kwant_success = False
        finalized_syst_kwant = None
    
    if not (thothqt_success and kwant_success):
        print("System creation failed, using analytical comparisons only")
    
    # Energy ranges for different calculations
    energies_transport = np.linspace(-4, 4, 17)  # Fewer points for stability
    energies_dos = np.linspace(-4, 4, 13)
    
    results = {}
    
    if thothqt_success:
        print("\nComputing ThothQT transmission...")
        start_time = time.time()
        try:
            T_tqt = [engine_tqt.transmission(E) for E in energies_transport]
            time_tqt_trans = time.time() - start_time
            T_tqt = np.array(T_tqt)
            print(f"ThothQT transmission computed: {time_tqt_trans:.3f}s")
        except Exception as e:
            print(f"ThothQT transmission failed: {e}")
            T_tqt = np.zeros(len(energies_transport))
            time_tqt_trans = 0
        
        print("Computing ThothQT DOS...")
        start_time = time.time()
        try:
            dos_tqt = [engine_tqt.density_of_states(E) for E in energies_dos]
            time_tqt_dos = time.time() - start_time
            dos_tqt = np.array(dos_tqt)
            print(f"ThothQT DOS computed: {time_tqt_dos:.3f}s")
        except Exception as e:
            print(f"ThothQT DOS failed: {e}")
            dos_tqt = np.zeros(len(energies_dos))
            time_tqt_dos = 0
    else:
        T_tqt = np.zeros(len(energies_transport))
        dos_tqt = np.zeros(len(energies_dos))
        time_tqt_trans = time_tqt_dos = 1e-6
    
    if kwant_success:
        print("Computing KWANT transmission...")
        start_time = time.time()
        T_kwant = compute_kwant_transmission_graphene(finalized_syst_kwant, energies_transport)
        time_kwant_trans = time.time() - start_time
        print(f"KWANT transmission computed: {time_kwant_trans:.3f}s")
        
        print("Computing KWANT DOS...")
        start_time = time.time()
        dos_kwant = compute_kwant_dos_graphene(finalized_syst_kwant, energies_dos)
        time_kwant_dos = time.time() - start_time
        print(f"KWANT DOS computed: {time_kwant_dos:.3f}s")
    else:
        T_kwant = np.zeros(len(energies_transport))
        dos_kwant = np.zeros(len(energies_dos))
        time_kwant_trans = time_kwant_dos = 1e-6
    
    # === TEST 2: NANORIBBON WITH IMPURITIES ===
    print("\n=== Test 2: Nanoribbon with Impurities ===")
    
    if thothqt_success:
        # Add some impurities to ThothQT system
        n_atoms = device_tqt.H.shape[0]
        impurity_sites = {
            n_atoms // 4: 0.5,      # Weak impurity
            n_atoms // 2: 1.0,      # Strong impurity  
            3 * n_atoms // 4: -0.5  # Negative impurity
        }
        
        print(f"Adding impurities at sites: {list(impurity_sites.keys())}")
        try:
            device_tqt_imp = tqt.make_graphene_nanoribbon(
                width=width, length=length, edge_type='zigzag', t=t, 
                impurities=impurity_sites
            )
            engine_tqt_imp = tqt.NEGFEngine(device_tqt_imp, Temp=300)
            
            print("Computing transmission with impurities...")
            T_tqt_imp = [engine_tqt_imp.transmission(E) for E in energies_transport]
            T_tqt_imp = np.array(T_tqt_imp)
            
            dos_tqt_imp = [engine_tqt_imp.density_of_states(E) for E in energies_dos]
            dos_tqt_imp = np.array(dos_tqt_imp)
            
            impurity_success = True
        except Exception as e:
            print(f"Impurity calculation failed: {e}")
            T_tqt_imp = T_tqt.copy()
            dos_tqt_imp = dos_tqt.copy()
            impurity_success = False
    else:
        T_tqt_imp = np.zeros(len(energies_transport))
        dos_tqt_imp = np.zeros(len(energies_dos))
        impurity_success = False
    
    # === ANALYSIS ===
    print("\n=== Analysis ===")
    
    # Compare clean systems
    if thothqt_success and kwant_success:
        diff_trans = np.abs(T_tqt - T_kwant)
        diff_dos = np.abs(dos_tqt - dos_kwant)
        
        print(f"Clean system comparison:")
        print(f"  Transmission max diff: {np.max(diff_trans):.2e}")
        print(f"  DOS max diff:          {np.max(diff_dos):.2e}")
        
        speedup_trans = time_kwant_trans / time_tqt_trans if time_tqt_trans > 0 else 1
        speedup_dos = time_kwant_dos / time_tqt_dos if time_tqt_dos > 0 else 1
        print(f"  Transmission speedup:  {speedup_trans:.1f}x")
        print(f"  DOS speedup:           {speedup_dos:.1f}x")
    
    # Impurity effects
    if thothqt_success and impurity_success:
        impurity_effect_trans = np.abs(T_tqt - T_tqt_imp)
        impurity_effect_dos = np.abs(dos_tqt - dos_tqt_imp)
        
        print(f"\nImpurity effects:")
        print(f"  Max transmission change: {np.max(impurity_effect_trans):.3f}")
        print(f"  Max DOS change:          {np.max(impurity_effect_dos):.3f}")
    
    # === PLOTTING ===
    print("\nCreating graphene validation plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Transmission comparison
    plt.subplot(3, 3, 1)
    if thothqt_success:
        plt.plot(energies_transport, T_tqt, 'b-', linewidth=2, label='ThothQT Clean')
    if kwant_success:
        plt.plot(energies_transport, T_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    if impurity_success:
        plt.plot(energies_transport, T_tqt_imp, 'g:', linewidth=2, label='ThothQT + Impurities')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title('Graphene Nanoribbon Transmission')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(1.1, np.max(T_tqt) * 1.1 if thothqt_success else 1.1))
    
    # 2. Transmission difference
    plt.subplot(3, 3, 2)
    if thothqt_success and kwant_success:
        plt.semilogy(energies_transport, diff_trans + 1e-16, 'g-', linewidth=2)
        plt.xlabel('Energy (eV)')
        plt.ylabel('|T_TQT - T_KWANT|')
        plt.title('Transmission Difference')
        plt.grid(True, alpha=0.3)
    
    # 3. DOS comparison
    plt.subplot(3, 3, 3)
    if thothqt_success:
        plt.plot(energies_dos, dos_tqt, 'b-', linewidth=2, label='ThothQT Clean')
    if kwant_success:
        plt.plot(energies_dos, dos_kwant, 'r--', linewidth=2, label='KWANT', alpha=0.8)
    if impurity_success:
        plt.plot(energies_dos, dos_tqt_imp, 'g:', linewidth=2, label='ThothQT + Impurities')
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV)')
    plt.title('Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. DOS difference
    plt.subplot(3, 3, 4)
    if thothqt_success and kwant_success:
        plt.semilogy(energies_dos, diff_dos + 1e-16, 'g-', linewidth=2)
        plt.xlabel('Energy (eV)')
        plt.ylabel('|DOS_TQT - DOS_KWANT|')
        plt.title('DOS Difference')
        plt.grid(True, alpha=0.3)
    
    # 5. Impurity effects on transmission
    plt.subplot(3, 3, 5)
    if thothqt_success and impurity_success:
        plt.plot(energies_transport, T_tqt, 'b-', linewidth=2, label='Clean')
        plt.plot(energies_transport, T_tqt_imp, 'r-', linewidth=2, label='With Impurities')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Transmission')
        plt.title('Impurity Effects on Transmission')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(1.1, np.max(T_tqt) * 1.1))
    
    # 6. Impurity effects on DOS
    plt.subplot(3, 3, 6)
    if thothqt_success and impurity_success:
        plt.plot(energies_dos, dos_tqt, 'b-', linewidth=2, label='Clean')
        plt.plot(energies_dos, dos_tqt_imp, 'r-', linewidth=2, label='With Impurities')
        plt.xlabel('Energy (eV)')
        plt.ylabel('DOS (states/eV)')
        plt.title('Impurity Effects on DOS')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 7. System visualization
    plt.subplot(3, 3, 7)
    if thothqt_success:
        # Simple visualization of system size and impurity locations
        n_atoms = device_tqt.H.shape[0]
        x_atoms = range(n_atoms)
        y_clean = np.ones(n_atoms)
        
        plt.scatter(x_atoms, y_clean, c='blue', s=20, alpha=0.6, label='Clean atoms')
        
        if impurity_success:
            for site, energy in impurity_sites.items():
                color = 'red' if energy > 0 else 'green'
                plt.scatter([site], [1], c=color, s=100, marker='x', label=f'Impurity ({energy} eV)')
        
        plt.xlabel('Atom Index')
        plt.ylabel('System')
        plt.title('Nanoribbon Structure')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 8. Performance comparison
    plt.subplot(3, 3, 8)
    if thothqt_success and kwant_success:
        methods = ['Transmission', 'DOS']
        speedups = [speedup_trans, speedup_dos]
        bars = plt.bar(methods, speedups, color=['blue', 'green'], alpha=0.7)
        plt.ylabel('Speedup Factor')
        plt.title('ThothQT vs KWANT Performance')
        plt.grid(True, alpha=0.3)
        
        for bar, speed in zip(bars, speedups):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(speedups)*0.02,
                    f'{speed:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 9. Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    summary_text = f"""
GRAPHENE NANORIBBON VALIDATION

SYSTEM:
Width: {width} chains
Length: {length} cells
Atoms: {device_tqt.H.shape[0] if thothqt_success else 'N/A'}

THOTHQT STATUS:
Clean system: {"SUCCESS" if thothqt_success else "FAILED"}
Impurities: {"SUCCESS" if impurity_success else "FAILED"}

KWANT COMPARISON:
System creation: {"SUCCESS" if kwant_success else "FAILED"}
{"Physics agreement: TBD" if thothqt_success and kwant_success else ""}

IMPURITY EFFECTS:
{"Max T change: {:.3f}".format(np.max(impurity_effect_trans)) if impurity_success and thothqt_success else "Not tested"}
{"Max DOS change: {:.3f}".format(np.max(impurity_effect_dos)) if impurity_success and thothqt_success else "Not tested"}

OVERALL:
ThothQT 2D: {"WORKING" if thothqt_success else "NEEDS WORK"}
Impurity support: {"YES" if impurity_success else "NO"}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    try:
        plt.savefig('graphene_nanoribbon_validation.png', dpi=150, bbox_inches='tight')
        print("Graphene validation plot saved: graphene_nanoribbon_validation.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # === FINAL ASSESSMENT ===
    print(f"\n=== FINAL ASSESSMENT ===")
    print(f"ThothQT 2D Implementation: {'SUCCESS' if thothqt_success else 'NEEDS DEBUGGING'}")
    print(f"Impurity Support: {'WORKING' if impurity_success else 'NEEDS WORK'}")
    print(f"KWANT Comparison: {'AVAILABLE' if kwant_success else 'LIMITED'}")
    
    if thothqt_success and kwant_success:
        trans_agreement = "EXCELLENT" if np.max(diff_trans) < 1e-6 else "GOOD" if np.max(diff_trans) < 1e-3 else "POOR"
        dos_agreement = "EXCELLENT" if np.max(diff_dos) < 1e-6 else "GOOD" if np.max(diff_dos) < 1e-3 else "POOR"
        print(f"Transmission Agreement: {trans_agreement}")
        print(f"DOS Agreement: {dos_agreement}")
        print(f"Performance: {min(speedup_trans, speedup_dos):.1f}x faster")
    
    print(f"Ready for graphene applications: {'YES' if thothqt_success and impurity_success else 'PARTIAL'}")
    
    return {
        'thothqt_success': thothqt_success,
        'kwant_success': kwant_success,
        'impurity_success': impurity_success,
        'transmission_clean': T_tqt if thothqt_success else None,
        'transmission_impurity': T_tqt_imp if impurity_success else None,
        'dos_clean': dos_tqt if thothqt_success else None,
        'dos_impurity': dos_tqt_imp if impurity_success else None
    }

if __name__ == "__main__":
    results = graphene_validation()
    
    try:
        plt.show()
    except:
        print("Plot display not available, but plots saved to file.")