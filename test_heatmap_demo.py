#!/usr/bin/env python3
"""
Test script to demonstrate the new sensitivity heat map functionality.

This script shows how to use the new compute_sensitivity_heatmap() and
plot_sensitivity_heatmap() functions to create temperature vs bias voltage
sensitivity maps.

Usage:
    python test_heatmap_demo.py --lanthanide Tb --use_scf
"""

import sys
import os
sys.path.append('parallel')

# Import the main module functions
from graphene_tb_Tb_SW_negf import (
    create_argparser, setup_system, example_sensitivity_heatmap_usage,
    compute_sensitivity_heatmap, plot_sensitivity_heatmap
)
import numpy as np
import matplotlib.pyplot as plt

def quick_heatmap_demo():
    """Minimal example showing heat map generation"""
    
    print("=== SENSITIVITY HEAT MAP DEMO ===")
    
    # Setup basic parameters  
    parser = create_argparser()
    args = parser.parse_args([
        '--lanthanide', 'Tb',
        '--length', '20', 
        '--width', '10',
        '--E', '-0.1',
        '--use_scf'  # Use SCF for realistic physics
    ])
    
    # Quick parameter check
    print(f"System: {args.lanthanide} in {args.length}×{args.width} graphene ribbon")
    print(f"Fermi energy: {args.E:.3f} eV")
    print(f"Using: {'SCF solver' if args.use_scf else 'Kwant S-matrix'}")
    
    # Build systems
    print("\nBuilding systems...")
    fsys_sw, fsys_pr = setup_system(args)
    
    # Define small parameter ranges for quick demo
    temperatures = np.array([50, 150, 250])  # Just 3 temperatures
    bias_voltages = np.array([-0.05, 0.0, 0.05])  # Just 3 bias points
    
    print(f"\nComputing {len(temperatures)}×{len(bias_voltages)} heat map...")
    print("(This is a reduced demo - use example_sensitivity_heatmap_usage() for full analysis)")
    
    # Compute heat map
    heatmap_data = compute_sensitivity_heatmap(
        fsys_sw=fsys_sw,
        fsys_pr=fsys_pr,
        args=args, 
        temperatures=temperatures,
        bias_voltages=bias_voltages
    )
    
    # Create simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_sensitivity_heatmap(heatmap_data, args, ax=ax, system_type='SW')
    plt.title(f'Quick Demo: {args.lanthanide} Sensitivity Heat Map')
    
    # Save plot
    filename = f"heatmap_demo_{args.lanthanide}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nDemo plot saved: {filename}")
    
    # Show statistics
    print(f"\nQuick Results:")
    print(f"SW sensitivity range: {np.min(heatmap_data['sensitivity_sw']):.2e} - {np.max(heatmap_data['sensitivity_sw']):.2e} S/X")
    if 'sensitivity_pr' in heatmap_data:
        enhancement = heatmap_data['sensitivity_enhancement']
        print(f"Enhancement factor: {np.min(enhancement):.1f}x - {np.max(enhancement):.1f}x")
    
    plt.show()
    return heatmap_data

def full_heatmap_demo():
    """Full example using the built-in demonstration function"""
    
    print("=== FULL SENSITIVITY HEAT MAP DEMO ===")
    
    # Setup parameters for comprehensive analysis
    parser = create_argparser()
    args = parser.parse_args([
        '--lanthanide', 'Tb',
        '--length', '24',
        '--width', '12', 
        '--E', '-0.05',
        '--use_scf',
        '--NX', '15'  # Field sweep points
    ])
    
    # Build systems
    print("Building systems for full analysis...")
    fsys_sw, fsys_pr = setup_system(args)
    
    # Use the built-in example function
    print("\nRunning comprehensive heat map analysis...")
    heatmap_data, fig = example_sensitivity_heatmap_usage(fsys_sw, fsys_pr, args)
    
    return heatmap_data, fig

if __name__ == "__main__":
    
    # Choose demo type
    import argparse
    parser = argparse.ArgumentParser(description='Heat map demo options')
    parser.add_argument('--full', action='store_true', 
                       help='Run full demo (slower but comprehensive)')
    demo_args = parser.parse_args()
    
    if demo_args.full:
        print("Running full comprehensive heat map demo...")
        heatmap_data, fig = full_heatmap_demo()
    else:
        print("Running quick heat map demo...")
        heatmap_data = quick_heatmap_demo()
    
    print("\n=== DEMO COMPLETE ===")
    print("To run full analysis, use: python test_heatmap_demo.py --full")
    print("Or call example_sensitivity_heatmap_usage() from your own script")