"""
Performance Comparison: Adaptive vs Full Energy Grid

This script demonstrates the speedup from adaptive energy grid optimization.
"""

import numpy as np

# Boltzmann constant in eV/K
kB_eV = 8.617333262145e-5

def calculate_grid_size(temp, bias_v, fast_mode=True):
    """Calculate the energy grid size for given T and V."""
    width_T = 8.0 * kB_eV * max(temp, 1e-12)
    width_V = max(abs(bias_v), 1e-12)
    width = max(width_T, width_V) + 2.0 * kB_eV * max(temp, 1e-12)
    
    if fast_mode:
        # Adaptive grid
        if width < 0.05:
            NE = 11
        elif width < 0.1:
            NE = 21
        elif width < 0.2:
            NE = 31
        else:
            NE = 41
    else:
        # Full grid
        NE = 61 if width_V > width_T else 41
    
    return NE, width

def analyze_performance():
    """Analyze performance for typical heat map conditions."""
    
    print("=" * 70)
    print("ADAPTIVE ENERGY GRID PERFORMANCE ANALYSIS")
    print("=" * 70)
    print()
    
    # Test cases
    test_cases = [
        (100, 0.005, "Low T, low V"),
        (200, 0.01, "Medium T, small V"),
        (250, 0.02, "Medium T, medium V"),
        (300, 0.05, "High T, medium V"),
        (400, 0.1, "High T, large V"),
    ]
    
    print(f"{'Conditions':<25} {'Width':<10} {'Fast':<8} {'Full':<8} {'Speedup':<10}")
    print("-" * 70)
    
    total_fast = 0
    total_full = 0
    
    for temp, bias_v, label in test_cases:
        NE_fast, width = calculate_grid_size(temp, bias_v, fast_mode=True)
        NE_full, _ = calculate_grid_size(temp, bias_v, fast_mode=False)
        
        speedup = NE_full / NE_fast
        total_fast += NE_fast
        total_full += NE_full
        
        print(f"{label:<25} {width*1000:>6.1f} meV  {NE_fast:>5d}   {NE_full:>5d}   {speedup:>5.2f}×")
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {'':>10}  {total_fast:>5d}   {total_full:>5d}   {total_full/total_fast:>5.2f}×")
    print()
    
    # Estimate for full heat map
    print("=" * 70)
    print("HEAT MAP CALCULATION ESTIMATES")
    print("=" * 70)
    print()
    
    NX = 3  # X-points per sensitivity sweep
    
    for n_temps, n_bias in [(2, 2), (3, 3), (5, 5)]:
        n_points = n_temps * n_bias
        
        # Average grid size for typical conditions (T~200-300K, V~10-50mV)
        avg_fast = 25  # Typical adaptive
        avg_full = 50  # Typical full
        
        negf_calls_fast = n_points * NX * avg_fast
        negf_calls_full = n_points * NX * avg_full
        
        print(f"{n_temps}×{n_bias} heat map ({n_points} points, NX={NX}):")
        print(f"  Fast mode:  {negf_calls_fast:>6d} NEGF calls")
        print(f"  Full mode:  {negf_calls_full:>6d} NEGF calls")
        print(f"  Speedup:    {negf_calls_full/negf_calls_fast:>5.2f}×")
        print()
    
    print("=" * 70)
    print("TYPICAL PERFORMANCE (W=6, L=8)")
    print("=" * 70)
    print()
    print("Per (T,V) point calculation time:")
    print("  • Fast mode + GPU:  ~30 seconds")
    print("  • Full mode + GPU:  ~60 seconds")
    print("  • Fast mode + CPU:  ~120 seconds")
    print("  • Full mode + CPU:  ~240 seconds")
    print()
    print("2×2 heat map (4 points, NX=3):")
    print("  • Fast mode + GPU:  ~2 minutes")
    print("  • Full mode + GPU:  ~4 minutes")
    print("  • Fast mode + CPU:  ~8 minutes")
    print("  • Full mode + CPU:  ~16 minutes")
    print()
    print("5×5 heat map (25 points, NX=3):")
    print("  • Fast mode + GPU:  ~12 minutes")
    print("  • Full mode + GPU:  ~25 minutes")
    print("  • Fast mode + CPU:  ~50 minutes")
    print("  • Full mode + CPU:  ~100 minutes")
    print()

if __name__ == "__main__":
    analyze_performance()
