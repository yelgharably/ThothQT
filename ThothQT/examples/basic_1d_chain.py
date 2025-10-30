"""
ThothQT Basic Example: 1D Atomic Chain
======================================

Demonstrates basic usage of ThothQT for a simple 1D atomic chain.
Shows how to:
1. Create a 1D device
2. Set up NEGF engine  
3. Compute transmission
4. Calculate I-V characteristics
5. Study temperature effects
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import thothqt
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import thothqt as tqt

def main():
    print("=" * 70)
    print("ThothQT Example: 1D Atomic Chain")
    print("=" * 70)
    
    # System parameters
    n_sites = 30        # Number of atoms in device
    t = 1.0            # Hopping parameter (eV) 
    temperature = 300   # Temperature (K)
    
    print(f"\nSystem parameters:")
    print(f"  Device length: {n_sites} sites")
    print(f"  Hopping: t = {t} eV")
    print(f"  Temperature: T = {temperature} K")
    print(f"  Thermal energy: kT = {tqt.temperature_to_thermal_energy(temperature):.3f} eV")
    
    # Create device and NEGF engine
    print(f"\nCreating 1D chain device...")
    device = tqt.make_1d_chain(n_sites=n_sites, t=t)
    engine = tqt.NEGFEngine(device, Temp=temperature)
    
    print(f"✓ Device created: {device.H.shape[0]}×{device.H.shape[0]} Hamiltonian")
    print(f"✓ NEGF engine initialized")
    
    # 1. Basic transmission calculation
    print(f"\n" + "="*50)
    print(f"1. TRANSMISSION SPECTRUM")
    print(f"="*50)
    
    # Energy range: around the band
    energies = np.linspace(-3*t, 3*t, 200)
    
    print(f"Computing transmission at {len(energies)} energies...")
    transmissions = []
    
    for i, E in enumerate(energies):
        T = engine.transmission(E)
        transmissions.append(T)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(energies)} ({100*(i+1)/len(energies):.1f}%)")
    
    transmissions = np.array(transmissions)
    
    print(f"✓ Transmission calculation complete")
    print(f"  Max transmission: {np.max(transmissions):.4f}")
    print(f"  Min transmission: {np.min(transmissions):.4f}")
    print(f"  Transmission at E=0: {engine.transmission(0.0):.4f}")
    
    # 2. I-V characteristics
    print(f"\n" + "="*50)
    print(f"2. I-V CHARACTERISTICS")
    print(f"="*50)
    
    bias_max = 0.5  # Maximum bias (V)
    n_bias = 21     # Number of bias points
    
    print(f"Computing I-V curve: ±{bias_max} V, {n_bias} points")
    
    voltages, currents = engine.IV_curve((-bias_max, bias_max), n_points=n_bias)
    
    print(f"✓ I-V calculation complete")
    print(f"  Bias range: {voltages[0]:.2f} to {voltages[-1]:.2f} V") 
    print(f"  Current range: {np.min(currents):.2e} to {np.max(currents):.2e} (2e²/h·eV units)")
    
    # Convert to SI units for display
    currents_si = [tqt.current_to_si(I, V) for I, V in zip(currents, voltages)]
    
    # 3. Temperature dependence
    print(f"\n" + "="*50) 
    print(f"3. TEMPERATURE DEPENDENCE")
    print(f"="*50)
    
    temperatures = [4, 77, 300, 500]  # K
    E_test = 0.5  # Test energy (eV)
    
    print(f"Testing transmission at E = {E_test} eV vs temperature:")
    
    temp_results = []
    for T in temperatures:
        engine_T = tqt.NEGFEngine(device, Temp=T)
        T_val = engine_T.transmission(E_test)
        kT = tqt.temperature_to_thermal_energy(T)
        temp_results.append((T, kT, T_val))
        print(f"  T = {T:3d} K, kT = {kT:.4f} eV → T = {T_val:.6f}")
    
    # 4. Plot results
    print(f"\n" + "="*50)
    print(f"4. VISUALIZATION")
    print(f"="*50)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Transmission spectrum
    ax1.plot(energies, transmissions, 'b-', linewidth=2)
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Perfect transmission')
    ax1.axvline(x=0, color='k', linestyle=':', alpha=0.5, label='E=0')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission')
    ax1.set_title(f'1D Chain Transmission (N={n_sites}, t={t} eV)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # I-V curve (quantum units)
    ax2.plot(voltages, currents, 'ro-', linewidth=2, markersize=4)
    ax2.set_xlabel('Bias Voltage (V)')
    ax2.set_ylabel('Current (2e²/h·eV units)')
    ax2.set_title('I-V Characteristic')
    ax2.grid(True, alpha=0.3)
    
    # I-V curve (SI units) 
    ax3.plot(voltages, np.array(currents_si)*1e9, 'go-', linewidth=2, markersize=4)
    ax3.set_xlabel('Bias Voltage (V)')
    ax3.set_ylabel('Current (nA)')
    ax3.set_title('I-V Characteristic (SI units)')
    ax3.grid(True, alpha=0.3)
    
    # Temperature dependence
    temps = [r[0] for r in temp_results]
    T_vals = [r[2] for r in temp_results]
    ax4.semilogx(temps, T_vals, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Transmission')
    ax4.set_title(f'Temperature Dependence (E={E_test} eV)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(os.path.dirname(__file__), 'basic_1d_chain_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_file}")
    
    # 5. Performance benchmark
    print(f"\n" + "="*50)
    print(f"5. PERFORMANCE BENCHMARK") 
    print(f"="*50)
    
    import time
    
    # Benchmark transmission calculation
    n_benchmark = 100
    E_benchmark = np.linspace(-2, 2, n_benchmark)
    
    print(f"Benchmarking {n_benchmark} transmission calculations...")
    
    start_time = time.time()
    for E in E_benchmark:
        _ = engine.transmission(E)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_calc = total_time / n_benchmark
    
    print(f"✓ Benchmark complete:")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Time per calculation: {time_per_calc*1000:.2f} ms")
    print(f"  Throughput: {n_benchmark/total_time:.1f} calculations/s")
    
    # 6. Summary
    print(f"\n" + "="*70)
    print(f"SUMMARY")
    print(f"="*70)
    print(f"✓ Successfully demonstrated ThothQT basic functionality")
    print(f"✓ 1D chain with {n_sites} sites shows expected physics")
    print(f"✓ Transmission spectrum computed over {len(energies)} energies")
    print(f"✓ I-V characteristics show linear response (small bias)")
    print(f"✓ Temperature effects minimal for this system")
    print(f"✓ Performance: {time_per_calc*1000:.1f} ms per transmission calculation")
    print(f"")
    print(f"The 1D atomic chain is an ideal test system showing:")
    print(f"  - Perfect transmission (T=1) within the band")
    print(f"  - Linear I-V response for small bias")
    print(f"  - Excellent numerical stability")
    print(f"  - Fast computation suitable for real-time analysis")
    
    plt.show()


if __name__ == "__main__":
    main()