#!/usr/bin/env python3
"""
Test script for new band structure and DOS functionality
Demonstrates the electronic structure analysis features added to the NEGF code
"""

import subprocess
import sys
import os

def test_electronic_structure_analysis():
    """Test the new electronic structure analysis functionality"""
    
    print("=== Testing Electronic Structure Analysis Functionality ===")
    print("This will run a quick analysis with the new band structure and DOS features")
    
    # Change to the parallel directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parallel_dir = os.path.join(script_dir, "parallel")
    
    if not os.path.exists(parallel_dir):
        print(f"Error: parallel directory not found at {parallel_dir}")
        return False
    
    script_path = os.path.join(parallel_dir, "graphene_tb_Tb_SW_negf.py")
    
    if not os.path.exists(script_path):
        print(f"Error: main script not found at {script_path}")
        return False
    
    # Test command with electronic structure analysis enabled
    test_cmd = [
        sys.executable, script_path,
        "--lanthanide", "Tb",
        "--system_size", "minimal",  # Fast test
        "--smoke",  # Quick test mode
        "--electronic_structure",  # Enable new functionality
        "--plot",  # Enable plotting
        "--save_json",  # Save results
        "--debug"  # Verbose output
    ]
    
    print(f"Running command: {' '.join(test_cmd)}")
    print(f"Working directory: {parallel_dir}")
    
    try:
        # Run the command
        result = subprocess.run(
            test_cmd,
            cwd=parallel_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("\n=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== STDERR ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Electronic structure analysis test completed successfully!")
            print("Look for the generated plots:")
            print("  - electronic_structure_Tb_*.png (band structure, DOS, wavefunctions)")
            print("  - sensitivity_analysis_*.png (sensitivity plots with placeholders)")
            return True
        else:
            print(f"\n✗ Test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        return False

def test_command_line_options():
    """Test that the new command line options are recognized"""
    
    print("\n=== Testing Command Line Options ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parallel_dir = os.path.join(script_dir, "parallel")
    script_path = os.path.join(parallel_dir, "graphene_tb_Tb_SW_negf.py")
    
    # Test help output to see if our new option is there
    help_cmd = [sys.executable, script_path, "--help"]
    
    try:
        result = subprocess.run(
            help_cmd,
            cwd=parallel_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "--electronic_structure" in result.stdout:
            print("✓ --electronic_structure option found in help")
            return True
        else:
            print("✗ --electronic_structure option not found in help")
            print("Help output:")
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"✗ Failed to check help: {e}")
        return False

if __name__ == "__main__":
    print("Testing new band structure and DOS functionality...")
    
    # Test 1: Check command line options
    options_ok = test_command_line_options()
    
    # Test 2: Run electronic structure analysis
    if options_ok:
        analysis_ok = test_electronic_structure_analysis()
        
        if analysis_ok:
            print("\n🎉 All tests passed! The new electronic structure functionality is working.")
            print("\nUsage examples:")
            print("  python graphene_tb_Tb_SW_negf.py --electronic_structure --lanthanide Tb")
            print("  python graphene_tb_Tb_SW_negf.py --electronic_structure --lanthanide Eu --system_size optimal")
        else:
            print("\n❌ Electronic structure analysis test failed")
    else:
        print("\n❌ Command line options test failed")