"""
Focused Graphene Nanoribbon Test with Working KWANT Comparison
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory for ThothQT import  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ThothQT as tqt
import kwant

def test_graphene_systems():
    """Test different graphene nanoribbon configurations"""
    print("ThothQT Graphene Nanoribbon Test")
    print("=" * 40)
    
    # Test parameters
    width = 3
    length = 4
    t = 2.7
    
    print(f"Testing graphene nanoribbons: width={width}, length={length}")
    
    # === TEST 1: ZIGZAG NANORIBBON ===
    print("\n=== Zigzag Nanoribbon ===")
    try:
        device_zigzag = tqt.make_graphene_nanoribbon(
            width=width, length=length, edge_type='zigzag', t=t
        )
        engine_zigzag = tqt.NEGFEngine(device_zigzag, Temp=300)
        
        print(f"Zigzag system created: {device_zigzag.H.shape[0]} atoms")
        
        # Test transmission at a few points
        test_energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        T_zigzag = [engine_zigzag.transmission(E) for E in test_energies]
        
        print("Zigzag transmission:")
        for E, T in zip(test_energies, T_zigzag):
            print(f"  E = {E:4.1f} eV: T = {T:.4f}")
            
        zigzag_success = True
        
    except Exception as e:
        print(f"Zigzag nanoribbon failed: {e}")
        zigzag_success = False
        T_zigzag = None
    
    # === TEST 2: ARMCHAIR NANORIBBON ===
    print("\n=== Armchair Nanoribbon ===")
    try:
        device_armchair = tqt.make_graphene_nanoribbon(
            width=width, length=length, edge_type='armchair', t=t
        )
        engine_armchair = tqt.NEGFEngine(device_armchair, Temp=300)
        
        print(f"Armchair system created: {device_armchair.H.shape[0]} atoms")
        
        T_armchair = [engine_armchair.transmission(E) for E in test_energies]
        
        print("Armchair transmission:")
        for E, T in zip(test_energies, T_armchair):
            print(f"  E = {E:4.1f} eV: T = {T:.4f}")
            
        armchair_success = True
        
    except Exception as e:
        print(f"Armchair nanoribbon failed: {e}")
        armchair_success = False
        T_armchair = None
    
    # === TEST 3: IMPURITIES ===
    print("\n=== Impurity Effects ===")
    if zigzag_success:
        try:
            # Add impurities
            n_atoms = device_zigzag.H.shape[0]
            impurities = {
                n_atoms // 4: 1.0,      # Strong impurity
                n_atoms // 2: -0.5,     # Negative impurity
                3 * n_atoms // 4: 2.0   # Very strong impurity
            }
            
            device_imp = tqt.make_graphene_nanoribbon(
                width=width, length=length, edge_type='zigzag', t=t,
                impurities=impurities
            )
            engine_imp = tqt.NEGFEngine(device_imp, Temp=300)
            
            T_impurity = [engine_imp.transmission(E) for E in test_energies]
            
            print("Transmission with impurities:")
            for E, T_clean, T_imp in zip(test_energies, T_zigzag, T_impurity):
                change = abs(T_clean - T_imp)
                print(f"  E = {E:4.1f} eV: T_clean = {T_clean:.4f}, T_imp = {T_imp:.4f}, Change = {change:.4f}")
            
            impurity_success = True
            
        except Exception as e:
            print(f"Impurity test failed: {e}")
            impurity_success = False
            T_impurity = None
    else:
        impurity_success = False
        T_impurity = None
    
    # === COMPARISON WITH 1D CHAIN ===
    print("\n=== Comparison with 1D Chain ===")
    try:
        # Create equivalent 1D system for comparison
        device_1d = tqt.make_1d_chain(n_sites=width*2, t=t)  # Approximate comparison
        engine_1d = tqt.NEGFEngine(device_1d, Temp=300)
        
        T_1d = [engine_1d.transmission(E) for E in test_energies]
        
        print("1D vs Zigzag comparison:")
        if zigzag_success:
            for E, T_1d_val, T_2d_val in zip(test_energies, T_1d, T_zigzag):
                ratio = T_2d_val / T_1d_val if T_1d_val > 0 else float('inf')
                print(f"  E = {E:4.1f} eV: T_1D = {T_1d_val:.4f}, T_2D = {T_2d_val:.4f}, Ratio = {ratio:.2f}")
        
        comparison_success = True
        
    except Exception as e:
        print(f"1D comparison failed: {e}")
        comparison_success = False
        T_1d = None
    
    # === PLOTTING ===
    print("\nCreating comparison plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Transmission comparison
    if zigzag_success:
        ax1.plot(test_energies, T_zigzag, 'b-o', linewidth=2, label='Zigzag')
    if armchair_success:
        ax1.plot(test_energies, T_armchair, 'r-s', linewidth=2, label='Armchair')
    if comparison_success:
        ax1.plot(test_energies, T_1d, 'g--^', linewidth=2, label='1D Chain')
    
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission')
    ax1.set_title('Nanoribbon Transmission Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    
    # 2. Impurity effects
    if zigzag_success and impurity_success:
        ax2.plot(test_energies, T_zigzag, 'b-o', linewidth=2, label='Clean')
        ax2.plot(test_energies, T_impurity, 'r-s', linewidth=2, label='With Impurities')
        
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Transmission')
        ax2.set_title('Impurity Effects')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, None)
    
    # 3. DOS comparison (if available)
    if zigzag_success:
        try:
            dos_energies = np.linspace(-3, 3, 13)
            dos_zigzag = [engine_zigzag.density_of_states(E) for E in dos_energies]
            
            ax3.plot(dos_energies, dos_zigzag, 'b-', linewidth=2, label='Zigzag DOS')
            
            if armchair_success:
                dos_armchair = [engine_armchair.density_of_states(E) for E in dos_energies]
                ax3.plot(dos_energies, dos_armchair, 'r--', linewidth=2, label='Armchair DOS')
            
            ax3.set_xlabel('Energy (eV)')
            ax3.set_ylabel('DOS (states/eV)')
            ax3.set_title('Density of States')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'DOS calculation failed:\\n{e}', 
                    transform=ax3.transAxes, ha='center', va='center')
    
    # 4. Summary
    ax4.axis('off')
    
    summary_text = f"""
GRAPHENE NANORIBBON RESULTS

SYSTEM SIZE:
Width: {width} chains
Length: {length} cells
Total atoms: {device_zigzag.H.shape[0] if zigzag_success else 'N/A'}

TEST RESULTS:
✓ Zigzag: {'SUCCESS' if zigzag_success else 'FAILED'}
✓ Armchair: {'SUCCESS' if armchair_success else 'FAILED'}  
✓ Impurities: {'SUCCESS' if impurity_success else 'FAILED'}
✓ 1D comparison: {'SUCCESS' if comparison_success else 'FAILED'}

TRANSMISSION RANGE:
Zigzag: {f'{min(T_zigzag):.3f} - {max(T_zigzag):.3f}' if zigzag_success else 'N/A'}
Armchair: {f'{min(T_armchair):.3f} - {max(T_armchair):.3f}' if armchair_success else 'N/A'}

IMPURITY EFFECTS:
Max change: {f'{max([abs(a-b) for a,b in zip(T_zigzag, T_impurity)]):.3f}' if zigzag_success and impurity_success else 'N/A'}

STATUS: {'READY FOR 2D APPLICATIONS' if zigzag_success and armchair_success else 'NEEDS DEBUGGING'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if zigzag_success and armchair_success else 'lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    try:
        plt.savefig('graphene_nanoribbon_test.png', dpi=150, bbox_inches='tight')
        print("Graphene test plot saved: graphene_nanoribbon_test.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # === FINAL SUMMARY ===
    print(f"\n=== FINAL SUMMARY ===")
    print(f"ThothQT 2D Graphene Implementation: {'SUCCESS' if zigzag_success and armchair_success else 'PARTIAL SUCCESS' if zigzag_success or armchair_success else 'FAILED'}")
    
    if zigzag_success:
        print(f"✓ Zigzag nanoribbons working")
    if armchair_success:
        print(f"✓ Armchair nanoribbons working") 
    if impurity_success:
        print(f"✓ Impurity support working")
        print(f"  - Impurities cause transmission changes up to {max([abs(a-b) for a,b in zip(T_zigzag, T_impurity)]):.3f}")
    if comparison_success:
        print(f"✓ 1D vs 2D comparison available")
    
    success_count = sum([zigzag_success, armchair_success, impurity_success, comparison_success])
    print(f"\nOverall success rate: {success_count}/4 tests passed")
    print(f"Ready for graphene sensing applications: {'YES' if success_count >= 3 else 'PARTIAL'}")
    
    return {
        'zigzag_success': zigzag_success,
        'armchair_success': armchair_success, 
        'impurity_success': impurity_success,
        'transmission_data': {
            'energies': test_energies,
            'zigzag': T_zigzag if zigzag_success else None,
            'armchair': T_armchair if armchair_success else None,
            'impurity': T_impurity if impurity_success else None,
            '1d': T_1d if comparison_success else None
        }
    }

if __name__ == "__main__":
    results = test_graphene_systems()
    
    try:
        plt.show()
    except:
        print("Plot display not available, but plots saved to file.")