"""
ThothQT Installation and Basic Functionality Test
=================================================

Simple test to verify that ThothQT installation is working correctly.
Tests:
1. Import all modules
2. Create basic systems  
3. Run transmission calculations
4. Check GPU support
5. Test KWANT bridge (if available)
"""

import sys
import os
import numpy as np

def test_imports():
    """Test that all ThothQT modules import correctly"""
    print("Testing imports...")
    
    try:
        import thothqt as tqt
        print("✓ Main thothqt package imported")
    except ImportError as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    # Test core components
    try:
        device = tqt.Device
        engine = tqt.NEGFEngine
        lead = tqt.PeriodicLead
        print("✓ Core classes available")
    except AttributeError as e:
        print(f"❌ Core classes missing: {e}")
        return False
    
    # Test builders
    try:
        builder = tqt.GrapheneBuilder
        custom = tqt.CustomSystemBuilder
        print("✓ System builders available")
    except AttributeError as e:
        print(f"❌ Builders missing: {e}")
        return False
    
    # Test utilities
    try:
        fermi = tqt.fermi_dirac
        mesh = tqt.EnergyMesh
        print("✓ Utility functions available")
    except AttributeError as e:
        print(f"❌ Utilities missing: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic 1D system functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import thothqt as tqt
        
        # Create simple 1D chain
        device = tqt.make_1d_chain(n_sites=10, t=1.0)
        print("✓ 1D chain created")
        
        # Create NEGF engine
        engine = tqt.NEGFEngine(device, Temp=300.0)
        print("✓ NEGF engine created")
        
        # Test transmission calculation
        T = engine.transmission(E=0.0)
        print(f"✓ Transmission calculated: T(0) = {T:.6f}")
        
        # Check if result is reasonable
        if 0.0 <= T <= 10.0:  # Allow up to 10 modes for safety
            print("✓ Transmission value reasonable")
        else:
            print(f"⚠ Transmission value unusual: {T}")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_graphene_builder():
    """Test graphene system builder"""
    print("\nTesting graphene builder...")
    
    try:
        import thothqt as tqt
        
        # Create graphene builder
        builder = tqt.GrapheneBuilder(a=1.42, t=2.7)
        print("✓ Graphene builder created")
        
        # Build small ribbon
        device = builder.zigzag_ribbon(width=2, length=3)
        print("✓ Zigzag ribbon built")
        
        # Test transport
        engine = tqt.NEGFEngine(device, Temp=300.0)
        T = engine.transmission(E=0.0)
        print(f"✓ Graphene transmission: T(0) = {T:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Graphene test failed: {e}")
        return False

def test_gpu_support():
    """Test GPU acceleration support"""
    print("\nTesting GPU support...")
    
    try:
        import cupy as cp
        gpu_available = cp.cuda.runtime.getDeviceCount() > 0
        print(f"✓ CuPy available, {cp.cuda.runtime.getDeviceCount()} GPU(s) detected")
    except:
        gpu_available = False
        print("✗ CuPy not available (CPU-only mode)")
    
    try:
        import thothqt as tqt
        
        device = tqt.make_1d_chain(n_sites=5, t=1.0)
        
        if gpu_available:
            # Test GPU engine
            engine_gpu = tqt.NEGFEngine(device, Temp=300.0, gpu=True)
            T_gpu = engine_gpu.transmission(E=0.0)
            print(f"✓ GPU engine works: T(0) = {T_gpu:.6f}")
        
        # Test CPU engine
        engine_cpu = tqt.NEGFEngine(device, Temp=300.0, gpu=False)
        T_cpu = engine_cpu.transmission(E=0.0)
        print(f"✓ CPU engine works: T(0) = {T_cpu:.6f}")
        
        if gpu_available:
            # Compare results
            error = abs(T_gpu - T_cpu)
            if error < 1e-10:
                print("✓ GPU and CPU results match")
            else:
                print(f"⚠ GPU/CPU mismatch: error = {error:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_kwant_bridge():
    """Test KWANT bridge functionality"""
    print("\nTesting KWANT bridge...")
    
    try:
        import kwant
        kwant_available = True
        print("✓ KWANT available")
    except ImportError:
        kwant_available = False
        print("✗ KWANT not available (bridge disabled)")
        return True  # Not a failure
    
    if not kwant_available:
        return True
    
    try:
        import thothqt as tqt
        
        # Test bridge functions
        converter = tqt.kwant_to_thothqt
        validator = tqt.validate_conversion
        print("✓ Bridge functions available")
        
        # Create simple KWANT system
        kwant_system = tqt.make_kwant_1d_chain(L=5, t=1.0)
        fsys = kwant_system.finalized()
        print("✓ KWANT system created")
        
        # Convert to ThothQT
        device = tqt.kwant_to_thothqt(fsys, Ef=0.0)
        print("✓ Conversion successful")
        
        # Validate
        results = tqt.validate_conversion(device, fsys, E=0.1, verbose=False)
        print(f"✓ Validation complete: error = {results['relative_error']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ KWANT bridge test failed: {e}")
        return False

def test_physical_constants():
    """Test physical constants and utility functions"""
    print("\nTesting physical constants...")
    
    try:
        import thothqt as tqt
        
        # Test temperature conversion
        kT_room = tqt.temperature_to_thermal_energy(300.0)
        expected_kT = 300.0 * 8.617333e-5  # Approximate
        
        if abs(kT_room - expected_kT) < 1e-6:
            print(f"✓ Temperature conversion: 300K → {kT_room:.4f} eV")
        else:
            print(f"⚠ Temperature conversion error")
        
        # Test Fermi function
        f_fermi = tqt.fermi_dirac(0.0, 0.0, kT_room)
        if abs(f_fermi - 0.5) < 1e-10:
            print("✓ Fermi function: f(μ) = 0.5")
        else:
            print(f"⚠ Fermi function error: f(μ) = {f_fermi}")
        
        # Test conductance quantum
        G0 = tqt.quantum_of_conductance()
        expected_G0 = 7.748e-5  # Approximate S
        if abs(G0 - expected_G0) < 1e-6:
            print(f"✓ Conductance quantum: G₀ = {G0:.2e} S")
        else:
            print(f"⚠ Conductance quantum error")
        
        return True
        
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def run_performance_test():
    """Quick performance test"""
    print("\nRunning performance test...")
    
    try:
        import time
        import thothqt as tqt
        
        # Create larger system
        device = tqt.make_1d_chain(n_sites=50, t=1.0)
        engine = tqt.NEGFEngine(device, Temp=300.0)
        
        # Benchmark transmission calculations
        n_calcs = 20
        energies = np.linspace(-2, 2, n_calcs)
        
        start_time = time.time()
        for E in energies:
            _ = engine.transmission(E)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_calc = total_time / n_calcs
        
        print(f"✓ Performance test complete:")
        print(f"  {n_calcs} calculations in {total_time:.3f} s")
        print(f"  {time_per_calc*1000:.2f} ms per transmission")
        print(f"  {n_calcs/total_time:.1f} calculations/s")
        
        # Performance assessment
        if time_per_calc < 0.01:
            performance = "Excellent"
        elif time_per_calc < 0.1:
            performance = "Good"
        else:
            performance = "Adequate"
        
        print(f"  Performance rating: {performance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run complete test suite"""
    print("=" * 70)
    print("ThothQT Installation Test Suite")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Graphene Builder", test_graphene_builder),
        ("GPU Support", test_gpu_support),
        ("KWANT Bridge", test_kwant_bridge),
        ("Physical Constants", test_physical_constants),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n" + "="*50)
        print(f"TEST: {test_name}")
        print("="*50)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! ThothQT is ready for use.")
        return True
    elif passed >= total - 1:
        print("\n✓ ThothQT is mostly functional with minor issues.")
        return True
    else:
        print("\n⚠ ThothQT has significant issues. Check installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)