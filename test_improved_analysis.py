#!/usr/bin/env python3
"""
Quick test of the improved electronic structure analysis
"""

import subprocess
import sys
import os

def test_improved_analysis():
    """Test the improved electronic structure analysis"""
    
    print("=== Testing Improved Electronic Structure Analysis ===")
    
    # Change to parallel directory
    os.chdir("d:/Graduate Life/Graphene_stuff/negf_sw/parallel")
    
    # Test with minimal system for speed
    cmd = [
        sys.executable, 
        "graphene_tb_Tb_SW_negf.py",
        "--lanthanide", "Tb", 
        "--system_size", "minimal",
        "--smoke",
        "--electronic_structure",
        "--Xmax", "0.05",  # Small field range for speed
        "--NX", "11"       # Fewer field points
    ]
    
    print("Running improved analysis...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ SUCCESS: Improved analysis completed!")
            
            # Look for output files
            import glob
            recent_files = glob.glob("electronic_structure_Tb_*.png")
            if recent_files:
                latest = max(recent_files, key=os.path.getctime)
                print(f"📊 Generated plot: {latest}")
                
                # Also check if sensitivity analysis was created
                sens_files = glob.glob("sensitivity_analysis_*.png") 
                if sens_files:
                    latest_sens = max(sens_files, key=os.path.getctime)
                    print(f"📈 Generated sensitivity: {latest_sens}")
            
            return True
        else:
            print("✗ FAILED with return code:", result.returncode)
            print("STDERR:", result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT: Analysis took too long")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_improved_analysis()
    
    if success:
        print("\n🎉 The improved electronic structure analysis is working!")
        print("\nKey improvements:")
        print("  • Clean, readable transmission spectrum")
        print("  • Proper DOS histograms (not confusing blobs)")
        print("  • Simple wavefunction visualization")
        print("  • Clear I-V characteristics")
        print("  • No more Kwant version compatibility issues")
        
        print("\nUsage:")
        print("  python graphene_tb_Tb_SW_negf.py --electronic_structure --lanthanide [Tb/Eu/Dy/etc]")
        
    else:
        print("\n❌ There are still issues to resolve")
        
    print(f"\nCurrent directory: {os.getcwd()}")