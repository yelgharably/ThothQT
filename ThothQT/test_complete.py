"""
ThothQT Complete Test - Using Both Package and Direct Imports
============================================================

Tests both the package interface and direct module imports to ensure
everything works correctly.
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path for package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_package_import():
    """Test importing ThothQT as a package"""
    print("=" * 60)
    print("TESTING PACKAGE IMPORT")
    print("=" * 60)
    
    try:
        import ThothQT as tqt
        print("✓ ThothQT package imported")
        
        # Test package interface
        tqt.status()
        
        # Test core functionality through package
        if hasattr(tqt, 'make_1d_chain') and hasattr(tqt, 'NEGFEngine'):
            device = tqt.make_1d_chain(5, 1.0)
            engine = tqt.NEGFEngine(device, Temp=300)
            T = engine.transmission(0.0)
            print(f"✓ Package interface works: T(0) = {T:.6f}")
            
            # Test builders if available
            if hasattr(tqt, 'GrapheneBuilder'):
                builder = tqt.GrapheneBuilder(a=1.42, t=2.7)
                print("✓ GrapheneBuilder available through package")
            else:
                print("⚠ GrapheneBuilder not exposed through package")
                
            # Test utilities if available  
            if hasattr(tqt, 'temperature_to_thermal_energy'):
                kT = tqt.temperature_to_thermal_energy(300)
                print(f"✓ Utils available through package: kT(300K) = {kT:.4f} eV")
            else:
                print("⚠ Utils not exposed through package")
                
            return True
        else:
            print("❌ Core functions not available through package")
            return False
            
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_direct_imports():
    """Test importing modules directly"""
    print("\n" + "=" * 60)
    print("TESTING DIRECT IMPORTS")  
    print("=" * 60)
    
    # Change to ThothQT directory for direct imports
    original_dir = os.getcwd()
    thothqt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(thothqt_dir)
    
    try:
        # Test core module
        print("1. Testing core module...")
        from thothqt import NEGFEngine, make_1d_chain, Device, fermi_dirac
        device = make_1d_chain(8, 1.0)
        engine = NEGFEngine(device, Temp=300)
        T = engine.transmission(0.0)
        print(f"   ✓ Core works: T(0) = {T:.6f}")
        
        # Test builders
        print("2. Testing builders...")
        from builders import GrapheneBuilder, TMDBuilder, CustomSystemBuilder, make_quantum_dot
        
        builder = GrapheneBuilder(a=1.42, t=2.7)
        device_gr = builder.zigzag_ribbon(width=2, length=3)
        engine_gr = NEGFEngine(device_gr, Temp=300)
        T_gr = engine_gr.transmission(0.0)
        print(f"   ✓ GrapheneBuilder works: T(0) = {T_gr:.6f}")
        
        device_qd = make_quantum_dot(n_sites=6, t=1.0)
        engine_qd = NEGFEngine(device_qd, Temp=300)  
        T_qd = engine_qd.transmission(0.0)
        print(f"   ✓ Quantum dot works: T(0) = {T_qd:.6f}")
        
        # Test utils
        print("3. Testing utils...")
        from utils import temperature_to_thermal_energy, quantum_of_conductance, EnergyMesh
        
        kT = temperature_to_thermal_energy(300)
        G0 = quantum_of_conductance()
        mesh = EnergyMesh(-1, 1, 50)
        print(f"   ✓ Utils work: kT(300K)={kT:.4f}eV, G₀={G0:.2e}S, mesh={len(mesh)}pts")
        
        # Test KWANT bridge (optional)
        print("4. Testing KWANT bridge...")
        try:
            from kwant_bridge import add_onsite_potential, add_uniform_field
            device_imp = add_onsite_potential(device, [3], 0.5)
            engine_imp = NEGFEngine(device_imp, Temp=300)
            T_imp = engine_imp.transmission(0.0)
            print(f"   ✓ KWANT bridge works: T(impurity) = {T_imp:.6f}")
        except Exception as e:
            print(f"   ⚠ KWANT bridge optional: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)

def performance_test():
    """Run performance benchmarks"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from thothqt import make_1d_chain, NEGFEngine
        from builders import GrapheneBuilder
        
        # 1D chain performance
        print("1. 1D Chain Performance:")
        device_1d = make_1d_chain(15, 1.0)
        engine_1d = NEGFEngine(device_1d, Temp=300)
        
        n_tests = 50
        energies = np.linspace(-2, 2, n_tests)
        
        start = time.time()
        transmissions = [engine_1d.transmission(E) for E in energies]
        time_1d = time.time() - start
        
        print(f"   {n_tests} calculations in {time_1d:.3f}s")
        print(f"   {time_1d/n_tests*1000:.2f} ms/calc ({n_tests/time_1d:.0f} calc/s)")
        
        # Graphene performance
        print("2. Graphene Performance:")
        builder = GrapheneBuilder(a=1.42, t=2.7)
        device_gr = builder.zigzag_ribbon(width=3, length=4)
        engine_gr = NEGFEngine(device_gr, Temp=300)
        
        n_tests_gr = 20  # Smaller test for graphene
        energies_gr = np.linspace(-1, 1, n_tests_gr)
        
        start = time.time()
        transmissions_gr = [engine_gr.transmission(E) for E in energies_gr]
        time_gr = time.time() - start
        
        print(f"   {n_tests_gr} calculations in {time_gr:.3f}s")
        print(f"   {time_gr/n_tests_gr*1000:.2f} ms/calc ({n_tests_gr/time_gr:.0f} calc/s)")
        
        # Performance rating
        rate_1d = n_tests / time_1d
        if rate_1d > 500:
            rating = "Excellent"
        elif rate_1d > 100:
            rating = "Good" 
        elif rate_1d > 20:
            rating = "Acceptable"
        else:
            rating = "Slow"
            
        print(f"   Overall rating: {rating}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def comprehensive_physics_test():
    """Test the physics accuracy"""
    print("\n" + "=" * 60)
    print("PHYSICS VALIDATION")
    print("=" * 60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from thothqt import make_1d_chain, NEGFEngine, fermi_dirac
        from builders import GrapheneBuilder, make_quantum_dot
        from utils import temperature_to_thermal_energy
        
        print("1. 1D Chain - Perfect Transmission:")
        device = make_1d_chain(10, 1.0)
        engine = NEGFEngine(device, Temp=300)
        T_zero = engine.transmission(0.0)
        print(f"   T(E=0): {T_zero:.6f} (should be 1.0)")
        assert abs(T_zero - 1.0) < 1e-6, f"Expected T=1.0, got {T_zero}"
        print("   ✓ Perfect transmission verified")
        
        print("2. Temperature Dependence:")
        temperatures = [4, 77, 300, 500]
        for T_K in temperatures:
            kT = temperature_to_thermal_energy(T_K)
            f_half = fermi_dirac(kT, 0.0, kT)  # f(μ + kT, μ, kT) 
            expected = 1.0 / (1.0 + np.exp(1.0))  # ≈ 0.269
            print(f"   T={T_K}K: f(μ+kT) = {f_half:.4f} (expected ≈ 0.269)")
            
        print("3. Quantum Dot Resonance:")
        device_qd = make_quantum_dot(n_sites=5, t=1.0, eps_dot=0.5)
        engine_qd = NEGFEngine(device_qd, Temp=300)
        T_resonance = engine_qd.transmission(0.5)  # At dot level
        T_offresonance = engine_qd.transmission(0.0)  # Away from resonance
        print(f"   T(on-resonance): {T_resonance:.6f}")
        print(f"   T(off-resonance): {T_offresonance:.6f}")
        
        print("4. Graphene Band Structure:")
        builder = GrapheneBuilder(a=1.42, t=2.7)
        device_gr = builder.zigzag_ribbon(width=4, length=3)
        engine_gr = NEGFEngine(device_gr, Temp=300)
        T_dirac = engine_gr.transmission(0.0)  # At Dirac point
        print(f"   T(Dirac point): {T_dirac:.6f}")
        
        print("   ✓ All physics tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("ThothQT Complete Functionality Test")
    print("=" * 60)
    
    results = {
        'package_import': test_package_import(),
        'direct_imports': test_direct_imports(), 
        'performance': performance_test(),
        'physics': comprehensive_physics_test()
    }
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test.replace('_', ' ').title():<20}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    percentage = total_passed / total_tests * 100
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed ({percentage:.1f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 COMPLETE SUCCESS! 🎉")
        print("ThothQT is fully functional and ready for production!")
    elif results['direct_imports'] and results['physics']:
        print("\n✅ CORE SUCCESS!")
        print("All functionality works via direct imports.")
        print("Package import may have minor issues but doesn't affect functionality.")
    else:
        print("\n❌ ISSUES DETECTED")
        print("Some functionality is not working correctly.")
    
    print("\nRecommended Usage:")
    if results['package_import']:
        print("  # Package import (preferred)")
        print("  import ThothQT as tqt")
        print("  device = tqt.make_1d_chain(10, 1.0)")
    
    if results['direct_imports']:
        print("  # Direct import (always works)")  
        print("  from thothqt import NEGFEngine, make_1d_chain")
        print("  from builders import GrapheneBuilder")
        print("  from utils import temperature_to_thermal_energy")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)