#!/usr/bin/env python3
"""
Demonstration script for literature-based lanthanide parameters in graphene NEGF simulations.

This script shows how to use the enhanced features:
1. Different lanthanide elements with literature parameters
2. Automatic bias voltage range determination
3. Literature-accurate Stone-Wales hopping parameters
4. Enhanced Stark effect modeling

Usage examples:
    python lanthanide_demo.py --show-all-lanthanides
    python lanthanide_demo.py --lanthanide Tb --auto-bias
    python lanthanide_demo.py --lanthanide Eu --bias-regime quantum_regime
    python lanthanide_demo.py --compare-lanthanides
"""

import subprocess
import sys
import json
from datetime import datetime

def run_simulation(lanthanide, bias_regime='linear', additional_args=None):
    """Run a single simulation with specified parameters."""
    
    cmd = [
        sys.executable, 
        'graphene_tb_Tb_SW_negf.py',
        '--lanthanide', lanthanide,
        '--bias_regime', bias_regime,
        '--auto_bias_range',
        '--plot',
        '--use_negf',  # Use NEGF for accurate bias-dependent transport
        '--parallel_bias',  # Enable parallel multi-bias computation
        '--save_json'
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n🔬 Running simulation: {lanthanide} in {bias_regime} regime")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Simulation completed successfully")
            return True
        else:
            print(f"❌ Simulation failed with error:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Simulation timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def show_lanthanide_info():
    """Display information about all available lanthanides."""
    
    cmd = [sys.executable, 'graphene_tb_Tb_SW_negf.py', '--list_lanthanides']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Failed to get lanthanide info: {e}")
        return False

def compare_bias_regimes(lanthanide='Tb'):
    """Compare different bias regimes for a single lanthanide."""
    
    regimes = ['linear', 'nonlinear_weak', 'nonlinear_strong', 'quantum_regime']
    
    print(f"\n📊 **Bias Regime Comparison for {lanthanide}³⁺**\n")
    
    results = {}
    for regime in regimes:
        print(f"Running {regime} regime...")
        success = run_simulation(lanthanide, regime, ['--sw_only'])  # SW only for speed
        results[regime] = success
        
    print(f"\n📋 **Summary:**")
    for regime, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {regime:20}: {status}")
    
    return results

def compare_lanthanides(bias_regime='linear'):
    """Compare transport properties across different lanthanides."""
    
    lanthanides = ['Tb', 'Nd', 'Eu', 'Dy', 'Er']
    
    print(f"\n🧪 **Lanthanide Comparison in {bias_regime} regime**\n")
    
    results = {}
    for ln in lanthanides:
        print(f"Running {ln}³⁺...")
        success = run_simulation(ln, bias_regime, ['--sw_only'])
        results[ln] = success
        
    print(f"\n📋 **Summary:**")
    for ln, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED" 
        print(f"   {ln}³⁺:     {status}")
    
    return results

def demonstrate_advanced_features():
    """Demonstrate advanced simulation features."""
    
    print("\n🚀 **Advanced Feature Demonstration**\n")
    
    # 1. High-accuracy Terbium simulation with SCF
    print("1. High-accuracy Tb³⁺ simulation with SCF solver:")
    tb_success = run_simulation('Tb', 'quantum_regime', [
        '--use_scf',
        '--finite_bias', 
        '--L', '40',  # Larger system
        '--W', '25'
    ])
    
    # 2. Non-magnetic Europium with strong field effects
    print("\n2. Non-magnetic Eu³⁺ with enhanced Stark effect:")
    eu_success = run_simulation('Eu', 'nonlinear_strong', [
        '--Xmax', '0.2',  # Stronger electric field
        '--NX', '61'      # Higher resolution
    ])
    
    # 3. Erbium for telecom applications
    print("\n3. Er³⁺ for quantum information applications:")
    er_success = run_simulation('Er', 'linear', [
        '--Temp', '4.0',   # Low temperature
        '--use_finite_T'   # Include thermal effects
    ])
    
    results = {
        'Tb_SCF': tb_success,
        'Eu_Stark': eu_success, 
        'Er_quantum': er_success
    }
    
    print(f"\n📋 **Advanced Features Summary:**")
    for feature, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {feature:15}: {status}")
    
    return results

def main():
    """Main demonstration function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Lanthanide NEGF Demo Script")
    parser.add_argument('--show-all-lanthanides', action='store_true',
                       help='Show information about all available lanthanides')
    parser.add_argument('--lanthanide', default='Tb',
                       choices=['Tb', 'Nd', 'Eu', 'Dy', 'Er'],
                       help='Run single lanthanide simulation')
    parser.add_argument('--bias-regime', default='linear',
                       choices=['linear', 'nonlinear_weak', 'nonlinear_strong', 'quantum_regime', 'high_field'],
                       help='Bias voltage regime to use')
    parser.add_argument('--auto-bias', action='store_true',
                       help='Use automatic bias range determination')
    parser.add_argument('--compare-bias-regimes', action='store_true',
                       help='Compare all bias regimes for one lanthanide')
    parser.add_argument('--compare-lanthanides', action='store_true', 
                       help='Compare all lanthanides in one regime')
    parser.add_argument('--advanced-demo', action='store_true',
                       help='Run advanced feature demonstration')
    parser.add_argument('--full-demo', action='store_true',
                       help='Run complete demonstration (takes ~30 minutes)')
    
    args = parser.parse_args()
    
    print("🧬 **Lanthanide-Enhanced Graphene NEGF Simulation Demo**")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.show_all_lanthanides:
        show_lanthanide_info()
    
    elif args.compare_bias_regimes:
        compare_bias_regimes(args.lanthanide)
    
    elif args.compare_lanthanides:
        compare_lanthanides(args.bias_regime)
    
    elif args.advanced_demo:
        demonstrate_advanced_features()
    
    elif args.full_demo:
        print("\n🎯 **Running Complete Demonstration Suite**\n")
        
        print("Step 1: Lanthanide information")
        show_lanthanide_info()
        
        print("\nStep 2: Bias regime comparison (Tb³⁺)")
        compare_bias_regimes('Tb')
        
        print("\nStep 3: Lanthanide comparison (linear regime)")
        compare_lanthanides('linear')
        
        print("\nStep 4: Advanced features")
        demonstrate_advanced_features()
        
        print("\n🎉 **Full demonstration completed!**")
    
    else:
        # Single simulation
        additional_args = []
        if args.auto_bias:
            additional_args.append('--auto_bias_range')
        
        run_simulation(args.lanthanide, args.bias_regime, additional_args)
    
    print(f"\n✨ Demo completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()