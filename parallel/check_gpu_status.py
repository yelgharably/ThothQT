# Quick GPU Usage Diagnostic
# Run this to verify GPU acceleration is working

import sys
import os

print("=" * 60)
print("GPU ACCELERATION DIAGNOSTIC")
print("=" * 60)

# 1. Check CuPy
print("\n1. CuPy Status:")
try:
    import cupy as cp
    print(f"   ✅ CuPy installed: {cp.__version__}")
    print(f"   ✅ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Quick GPU test
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    print(f"   ✅ GPU compute test: {c.get()} (passed)")
    
    # Memory info
    mempool = cp.get_default_memory_pool()
    print(f"   ✅ GPU memory pool: {mempool.used_bytes() / 1024**2:.1f} MB used")
    
except ImportError as e:
    print(f"   ❌ CuPy NOT available: {e}")
    print("   → GPU acceleration DISABLED")
except Exception as e:
    print(f"   ⚠️  CuPy error: {e}")

# 2. Check GPU NEGF Solver
print("\n2. GPU NEGF Solver Status:")
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from gpu_negf_solver import GPUNEGFSolver, GPU_AVAILABLE
    
    print(f"   ✅ GPUNEGFSolver module imported")
    print(f"   ✅ GPU_AVAILABLE flag: {GPU_AVAILABLE}")
    
    if GPU_AVAILABLE:
        print("   ✅ GPU NEGF acceleration: ENABLED")
    else:
        print("   ❌ GPU NEGF acceleration: DISABLED (CuPy issue)")
        
except ImportError as e:
    print(f"   ❌ GPU solver NOT available: {e}")
except Exception as e:
    print(f"   ⚠️  Import error: {e}")

# 3. Check NEGF Core
print("\n3. NEGF Core Status:")
try:
    from negf_core import NEGFSolver
    print(f"   ✅ NEGFSolver (CPU fallback) available")
except ImportError as e:
    print(f"   ❌ NEGFSolver NOT available: {e}")

# 4. Check Poisson Solver
print("\n4. Poisson Solver GPU Status:")
try:
    from poisson_solver import PoissonSolver1D, GPU_AVAILABLE as POISSON_GPU
    print(f"   ✅ PoissonSolver imported")
    print(f"   ✅ Poisson GPU_AVAILABLE: {POISSON_GPU}")
except ImportError as e:
    print(f"   ❌ PoissonSolver NOT available: {e}")

# 5. Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

try:
    import cupy as cp
    from gpu_negf_solver import GPU_AVAILABLE as GPU_NEGF
    from poisson_solver import GPU_AVAILABLE as GPU_POISSON
    
    if GPU_NEGF and GPU_POISSON:
        print("✅ STATUS: FULL GPU ACCELERATION ACTIVE")
        print("   • NEGF transmission: GPU (CuPy)")
        print("   • Poisson solver: GPU (CuPy)")
        print("   • Expected speedup: 5-15× over CPU")
    elif GPU_NEGF:
        print("⚠️  STATUS: PARTIAL GPU (NEGF only)")
        print("   • NEGF transmission: GPU (CuPy)")
        print("   • Poisson solver: CPU fallback")
    elif GPU_POISSON:
        print("⚠️  STATUS: PARTIAL GPU (Poisson only)")
        print("   • NEGF transmission: CPU fallback")
        print("   • Poisson solver: GPU (CuPy)")
    else:
        print("❌ STATUS: CPU ONLY")
        print("   • All operations running on CPU")
        print("   • Install CuPy for GPU acceleration")
except:
    print("❌ STATUS: GPU UNAVAILABLE")
    print("   • CuPy not properly installed")
    print("   • Install: conda install -c conda-forge cupy")

print("=" * 60)
