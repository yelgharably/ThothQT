"""
Test the corrected GrapheneBuilder implementation
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add current directory for imports
sys.path.insert(0, '.')

# Import the classes from the fixed script
exec(open('thothqt_fixed.py').read())

def test_graphene_structure():
    """Test the graphene structure construction"""
    print("Testing GrapheneBuilder structure...")
    
    # Create small graphene nanoribbon
    gb = GrapheneBuilder(width=2, length=3, t=-2.7, edge_type='zigzag')
    
    print(f"Graphene nanoribbon created:")
    print(f"  Total atoms: {gb.H.shape[0]}")
    print(f"  Hamiltonian shape: {gb.H.shape}")
    print(f"  H01 shape: {gb.H01.shape}")
    print(f"  Non-zero elements in H: {gb.H.nnz}")
    
    # Check Hamiltonian properties
    H_dense = gb.H.toarray()
    print(f"  Is Hermitian: {np.allclose(H_dense, H_dense.conj().T)}")
    print(f"  Hopping values: {np.unique(H_dense[H_dense != 0])}")
    
    # Quick eigenvalue check
    eigenvals = np.linalg.eigvals(H_dense)
    eigenvals = np.sort(eigenvals)
    print(f"  Energy range: {eigenvals[0]:.3f} to {eigenvals[-1]:.3f} eV")
    
    # Test transmission at a few points
    left = PeriodicLead(gb.H01, gb.H01, gb.tau, is_right=False)
    right = PeriodicLead(gb.H01, gb.H01, gb.tau, is_right=True)
    engine = NEGFEngine(gb.H, left, right, use_gpu=False)  # Use CPU for debugging
    
    test_energies = [0.0, 1.0, -1.0]
    print("\nTransmission test:")
    for E in test_energies:
        try:
            T = engine.transmission(E)
            print(f"  T({E:+.1f} eV) = {T:.6f}")
        except Exception as e:
            print(f"  T({E:+.1f} eV) = ERROR: {e}")
    
    return gb, engine

def plot_simple_graphene():
    """Create a simple graphene transmission plot"""
    print("\nCreating graphene transmission plot...")
    
    gb, engine = test_graphene_structure()
    
    # Test with smaller energy range and fewer points
    energies = np.linspace(-3, 3, 20)
    Tvals = []
    
    for i, E in enumerate(energies):
        try:
            T = engine.transmission(E)
            Tvals.append(T)
            print(f"  E={E:.2f}: T={T:.6f}")
        except Exception as e:
            print(f"  E={E:.2f}: ERROR - {e}")
            Tvals.append(0.0)
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(energies, Tvals, 'bo-', linewidth=2, markersize=6)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Transmission")
    plt.title("Corrected Graphene Transmission")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(energies, np.array(Tvals) + 1e-12, 'ro-', linewidth=2, markersize=4)
    plt.xlabel("Energy (eV)")  
    plt.ylabel("log(Transmission)")
    plt.title("Log Scale")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("validation_results/corrected_graphene_test.png", dpi=150, bbox_inches='tight')
    print("Plot saved: validation_results/corrected_graphene_test.png")
    
    return energies, Tvals

if __name__ == "__main__":
    try:
        energies, Tvals = plot_simple_graphene()
        
        print(f"\n=== Summary ===")
        print(f"Transmission range: {min(Tvals):.6f} to {max(Tvals):.6f}")
        print(f"Non-zero transmissions: {sum(1 for T in Tvals if T > 1e-10)}/{len(Tvals)}")
        
        if max(Tvals) > 0.01:
            print("✓ Transmission values look reasonable!")
        else:
            print("⚠ Transmission values still very small")
            
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()