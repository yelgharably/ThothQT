# Systematic Investigation Protocol for Lanthanide-Doped Graphene NEGF Studies

## 🎯 **Research Objectives & Protocol Overview**

This protocol provides a systematic approach to investigate **lanthanide-doped graphene quantum transport** with proper physical behavior. The calculations are organized by scientific question and increasing complexity.

---

## 📋 **Phase 1: System Validation & Baseline Studies**

### **1.1 Parameter Validation** 
*Goal: Verify literature parameters and establish baseline behavior*

```bash
# Test all lanthanide elements with optimal parameters
for ln in Tb Nd Eu Dy Er; do
    echo "Testing $ln parameters..."
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --system_size optimal \
        --bias_regime linear \
        --plot \
        --save_json \
        --NX 21 --Xmax 0.05 \
        --sw_only
done
```

**Expected outcomes:**
- Validate literature α₁, α₂ parameters for each lanthanide
- Establish baseline Stark effect responses
- Verify system convergence and stability

### **1.2 Stone-Wales Defect Characterization**
*Goal: Quantify SW defect impact on transport properties*

```bash
# Compare SW vs Pristine systems
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --system_size optimal \
    --compare_sw \
    --bias_regime linear \
    --use_negf \
    --plot --save_json \
    --NX 41 --Xmax 0.1

# Test SW hopping parameter sensitivity
for hop_scale in 0.85 0.88 0.92 0.95; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --hop_scale_eta2 $hop_scale \
        --sw_only \
        --bias_regime linear \
        --save_json
done
```

**Expected outcomes:**
- Quantify conductance enhancement/suppression due to SW defects
- Validate literature hopping parameter modifications (0.85-0.95×t)
- Establish defect-enhanced Stark effect scaling

### **1.3 System Size Convergence Study**
*Goal: Ensure results are converged with respect to system dimensions*

```bash
# Test system size convergence
for size in minimal optimal large; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --system_size $size \
        --bias_regime linear \
        --sw_only \
        --save_json
done

# Manual size testing for critical dimensions
for W in 8 12 16 20 25; do
    for L in 20 30 40 50; do
        python graphene_tb_Tb_SW_negf.py \
            --lanthanide Tb \
            --W $W --L $L \
            --bias_regime linear \
            --sw_only \
            --save_json
    done
done
```

**Expected outcomes:**
- Identify minimum system size for converged results
- Establish computational efficiency vs accuracy trade-offs
- Validate literature-based size recommendations

---

## 🔬 **Phase 2: Physical Phenomena Investigation**

### **2.1 Stark Effect Characterization**
*Goal: Systematic study of electric field response for each lanthanide*

```bash
# High-resolution Stark effect mapping
for ln in Tb Nd Eu Dy Er; do
    echo "Stark effect study: $ln"
    
    # Linear regime (small fields)
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --system_size optimal \
        --NX 81 --Xmax 0.02 \
        --bias_regime linear \
        --plot --save_json
    
    # Nonlinear regime (strong fields) 
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --system_size optimal \
        --NX 101 --Xmax 0.2 \
        --bias_regime nonlinear_strong \
        --plot --save_json
done

# Special focus on Eu³⁺ (J=0 state)
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Eu \
    --system_size large \
    --NX 151 --Xmax 0.3 \
    --bias_regime high_field \
    --use_scf \
    --plot --save_json
```

**Expected outcomes:**
- Linear Stark coefficients α₁ for each lanthanide
- Quadratic coefficients α₂ and onset of nonlinearity
- J-state mixing effects (especially for Eu³⁺)
- Field-induced level crossings and anti-crossings

### **2.2 Temperature-Dependent Transport**
*Goal: Study thermal effects and quantum coherence*

```bash
# Temperature sweep for coherence length studies
for T in 4.0 10.0 20.0 77.0 150.0 300.0; do
    echo "Temperature study: $T K"
    
    # Er³⁺ for quantum coherence applications
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Er \
        --Temp $T \
        --use_finite_T \
        --bias_regime linear \
        --system_size optimal \
        --save_json
    
    # Tb³⁺ for magnetic applications  
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --Temp $T \
        --use_finite_T \
        --bias_regime linear \
        --system_size optimal \
        --save_json
done

# Low-temperature high-precision studies
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Er \
    --Temp 4.0 \
    --use_finite_T \
    --system_size large \
    --bias_regime linear \
    --NX 101 --Xmax 0.05 \
    --use_scf \
    --plot --save_json
```

**Expected outcomes:**
- Temperature dependence of conductance and sensitivity
- Thermal broadening effects on lanthanide resonances
- Coherence length scaling with temperature
- Optimal operating temperatures for each application

### **2.3 Magnetic Field Effects** 
*Goal: Investigate Zeeman splitting and magnetic response*

```bash
# Zeeman effect studies
for ln in Tb Nd Dy Er; do  # Skip Eu (J=0)
    echo "Magnetic field study: $ln"
    
    # B-field sweep
    for Bz in 0.0 0.1 0.2 0.5 1.0 2.0 5.0; do
        python graphene_tb_Tb_SW_negf.py \
            --lanthanide $ln \
            --Bz $Bz \
            --bias_regime linear \
            --system_size optimal \
            --save_json
    done
done

# High-field studies for magnetic switching
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --Bz 10.0 \
    --bias_regime quantum_regime \
    --system_size optimal \
    --use_negf \
    --plot --save_json

# Compare with non-magnetic Eu³⁺
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Eu \
    --Bz 5.0 \
    --bias_regime linear \
    --system_size optimal \
    --save_json
```

**Expected outcomes:**
- Zeeman splitting energies and g-factors
- Magnetic field tuning of transport properties  
- Comparison of magnetic vs non-magnetic lanthanides
- Potential for magnetic switching applications

---

## ⚡ **Phase 3: Finite-Bias & Nonlinear Transport**

### **3.1 Current-Voltage Characteristics**
*Goal: Study nonlinear transport and bias-dependent physics*

```bash
# Multi-bias I-V curves for all lanthanides
for ln in Tb Nd Eu Dy Er; do
    echo "I-V characteristics: $ln"
    
    # Linear response regime
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --bias_regime linear \
        --parallel_bias \
        --use_negf \
        --finite_bias \
        --plot --save_json
    
    # Extended nonlinear regime
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --bias_regime quantum_regime \
        --parallel_bias \
        --use_negf \
        --finite_bias \
        --plot --save_json
done

# High-resolution I-V for resonant tunneling
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --bias_voltages "$(python -c 'import numpy as np; print(\",\".join([f\"{v:.6f}\" for v in np.linspace(-0.1, 0.1, 201)]))')" \
    --parallel_bias \
    --use_negf \
    --finite_bias \
    --plot --save_json
```

**Expected outcomes:**
- Nonlinear I-V characteristics and differential conductance
- Resonant tunneling through 4f states
- Bias-dependent Stark effect and level alignment
- Quantum interference effects in finite bias

### **3.2 Self-Consistent Field Studies** 
*Goal: Include electrostatic effects for realistic device physics*

```bash
# SCF convergence for different bias regimes
for bias in 0.001 0.005 0.01 0.02 0.05 0.1; do
    echo "SCF study: bias = $bias V"
    
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --use_scf \
        --finite_bias \
        --bias_voltage $bias \
        --system_size optimal \
        --scf_tolerance 1e-6 \
        --save_json
done

# High-precision SCF for publication results
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --use_scf \
    --finite_bias \
    --bias_voltage 0.01 \
    --system_size large \
    --scf_tolerance 1e-7 \
    --scf_max_iters 100 \
    --parallel_bias \
    --plot --save_json

# SCF with strong electric fields
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Eu \
    --use_scf \
    --finite_bias \
    --bias_voltage 0.05 \
    --NX 51 --Xmax 0.3 \
    --scf_tolerance 1e-6 \
    --plot --save_json
```

**Expected outcomes:**
- Self-consistent electrostatic potentials
- Bias-dependent charge redistribution
- Convergence behavior in different transport regimes
- Validation of SCF vs linear response approximations

---

## 🎯 **Phase 4: Comparative & Optimization Studies**

### **4.1 Lanthanide Element Comparison**
*Goal: Systematic comparison across the lanthanide series*

```bash
# Comprehensive lanthanide comparison
echo "# Lanthanide Comparison Study" > lanthanide_comparison.sh
echo "# Generated on $(date)" >> lanthanide_comparison.sh

# Standard conditions for fair comparison
STANDARD_ARGS="--system_size optimal --bias_regime linear --use_negf --plot --save_json"

for ln in Tb Nd Eu Dy Er; do
    echo "# $ln studies" >> lanthanide_comparison.sh
    
    # Baseline transport
    echo "python graphene_tb_Tb_SW_negf.py --lanthanide $ln $STANDARD_ARGS" >> lanthanide_comparison.sh
    
    # Stark effect sensitivity
    echo "python graphene_tb_Tb_SW_negf.py --lanthanide $ln --NX 61 --Xmax 0.1 $STANDARD_ARGS" >> lanthanide_comparison.sh
    
    # Temperature dependence
    echo "python graphene_tb_Tb_SW_negf.py --lanthanide $ln --Temp 77 --use_finite_T $STANDARD_ARGS" >> lanthanide_comparison.sh
done

chmod +x lanthanide_comparison.sh
# ./lanthanide_comparison.sh
```

**Expected outcomes:**
- Relative Stark effect strengths (α₁, α₂ values)
- Transport efficiency comparison
- Temperature stability ranking
- Application-specific recommendations

### **4.2 Defect Engineering Studies**
*Goal: Optimize SW defect parameters for enhanced sensitivity*

```bash
# SW defect parameter optimization
for dE_on in 0.01 0.02 0.03 0.05 0.08; do
    for hop_scale in 0.85 0.88 0.92 0.95; do
        python graphene_tb_Tb_SW_negf.py \
            --lanthanide Tb \
            --dE_on_eta2 $dE_on \
            --hop_scale_eta2 $hop_scale \
            --bias_regime linear \
            --sw_only \
            --save_json
    done
done

# Coupling strength optimization
for V_eta2 in 0.5 0.6 0.65 0.7 0.8; do
    for V_side in 0.1 0.15 0.2 0.25 0.3; do
        python graphene_tb_Tb_SW_negf.py \
            --lanthanide Tb \
            --Vimp_eta2 $V_eta2 \
            --Vimp_side $V_side \
            --bias_regime linear \
            --sw_only \
            --save_json
    done
done
```

**Expected outcomes:**
- Optimal defect parameters for maximum sensitivity
- Trade-offs between coupling strength and transport efficiency
- Guidelines for experimental defect engineering

---

## 📊 **Phase 5: Application-Specific Protocols**

### **5.1 Quantum Sensing Applications**
*Goal: Optimize parameters for electric field sensing*

```bash
# Ultra-high sensitivity configuration
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --system_size large \
    --bias_regime linear \
    --NX 201 --Xmax 0.01 \
    --use_scf \
    --finite_bias \
    --Temp 4.0 \
    --use_finite_T \
    --plot --save_json

# Noise analysis and shot noise studies  
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Er \
    --system_size optimal \
    --bias_regime linear \
    --parallel_bias \
    --Temp 4.0 \
    --use_finite_T \
    --plot --save_json

# Compare sensing performance across lanthanides
for ln in Tb Nd Eu Dy Er; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --system_size optimal \
        --NX 101 --Xmax 0.02 \
        --bias_regime linear \
        --Temp 77 \
        --use_finite_T \
        --save_json
done
```

### **5.2 Quantum Information Processing**
*Goal: Study coherence and quantum state manipulation*

```bash
# Coherence studies at telecom wavelengths (Er³⁺)
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Er \
    --system_size large \
    --Temp 4.0 \
    --use_finite_T \
    --bias_regime linear \
    --NX 151 --Xmax 0.03 \
    --use_scf \
    --plot --save_json

# Magnetic switching for qubits (Tb³⁺)
for Bz in 0.0 0.5 1.0 1.5 2.0; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --Bz $Bz \
        --Temp 4.0 \
        --use_finite_T \
        --bias_regime linear \
        --system_size optimal \
        --save_json
done

# Non-magnetic qubit alternative (Eu³⁺)
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Eu \
    --system_size large \
    --Temp 4.0 \
    --use_finite_T \
    --NX 101 --Xmax 0.1 \
    --bias_regime nonlinear_strong \
    --use_scf \
    --plot --save_json
```

### **5.3 Room Temperature Applications**
*Goal: Investigate practical device performance*

```bash
# Room temperature transport optimization
for ln in Tb Nd Eu Dy Er; do
    # Standard conditions
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --Temp 300 \
        --system_size optimal \
        --bias_regime linear \
        --use_negf \
        --save_json
    
    # Enhanced coupling for room temperature operation
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide $ln \
        --Temp 300 \
        --Vimp_eta2 0.8 \
        --system_size optimal \
        --bias_regime nonlinear_weak \
        --use_negf \
        --save_json
done

# Thermal stability testing
for T in 250 300 350 400; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --Temp $T \
        --use_finite_T \
        --system_size optimal \
        --bias_regime linear \
        --save_json
done
```

---

## 🔧 **Phase 6: Method Validation & Benchmarking**

### **6.1 Convergence Testing**
*Goal: Validate numerical accuracy and establish error bars*

```bash
# Energy grid convergence
for NE in 21 51 101 201; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --bias_regime quantum_regime \
        --use_negf \
        --finite_bias \
        --save_json
done

# SCF tolerance convergence
for tol in 1e-4 1e-5 1e-6 1e-7; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --use_scf \
        --scf_tolerance $tol \
        --bias_voltage 0.01 \
        --save_json
done

# Field point convergence
for NX in 21 41 81 161; do
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --NX $NX \
        --Xmax 0.1 \
        --bias_regime linear \
        --save_json
done
```

### **6.2 Method Comparison**
*Goal: Compare different computational approaches*

```bash
# Compare Kwant vs NEGF vs SCF
for method in "" "--use_negf" "--use_scf --finite_bias"; do
    echo "Method: $method"
    python graphene_tb_Tb_SW_negf.py \
        --lanthanide Tb \
        --system_size optimal \
        --bias_regime linear \
        $method \
        --save_json
done

# Parallel vs serial bias calculations
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --bias_regime quantum_regime \
    --parallel_bias \
    --max_workers 4 \
    --use_negf \
    --save_json

# Large system validation
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Tb \
    --large_system \
    --system_size large \
    --use_scf \
    --bias_voltage 0.005 \
    --save_json
```

---

## 📈 **Data Analysis & Results Processing**

### **Post-Processing Scripts**

```bash
# Create analysis script
cat > analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze lanthanide NEGF simulation results.
Process JSON files and extract key physics parameters.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime

def extract_sensitivity(json_file):
    """Extract maximum dG/dX from results."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract sensitivity data
    if 'sw_system' in data:
        dGdX = data['sw_system'].get('dGdX', [])
        if dGdX:
            return np.max(np.abs(dGdX))
    return 0.0

def compare_lanthanides():
    """Compare transport properties across lanthanide series."""
    
    lanthanides = ['Tb', 'Nd', 'Eu', 'Dy', 'Er']
    results = {}
    
    for ln in lanthanides:
        # Find JSON files for this lanthanide
        pattern = f"*{ln}*.json"
        files = glob.glob(pattern)
        
        sensitivities = []
        for file in files:
            try:
                sens = extract_sensitivity(file)
                if sens > 0:
                    sensitivities.append(sens)
            except:
                continue
        
        if sensitivities:
            results[ln] = {
                'mean_sensitivity': np.mean(sensitivities),
                'std_sensitivity': np.std(sensitivities),
                'max_sensitivity': np.max(sensitivities),
                'n_calculations': len(sensitivities)
            }
    
    # Print summary
    print("Lanthanide Sensitivity Comparison:")
    print("Element | Mean Sens. | Max Sens. | Std Dev | N_calc")
    print("-" * 55)
    for ln, data in results.items():
        print(f"{ln:7} | {data['mean_sensitivity']:.2e} | {data['max_sensitivity']:.2e} | {data['std_sensitivity']:.2e} | {data['n_calculations']:6d}")
    
    return results

if __name__ == "__main__":
    results = compare_lanthanides()
EOF

chmod +x analyze_results.py
```

### **Visualization Scripts**

```bash
# Create plotting script
cat > plot_results.py << 'EOF'
#!/usr/bin/env python3
"""
Generate publication-quality plots from lanthanide NEGF results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_stark_effect_comparison():
    """Plot Stark effect comparison across lanthanides."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    lanthanides = ['Tb', 'Nd', 'Eu', 'Dy', 'Er']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (ln, color) in enumerate(zip(lanthanides, colors)):
        ax = axes[i]
        
        # Find data files for this lanthanide
        pattern = f"*{ln}*linear*.json"
        files = glob.glob(pattern)
        
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                if 'sw_system' in data:
                    X_vals = data['sw_system'].get('X_values', [])
                    G_vals = data['sw_system'].get('conductances', [])
                    
                    if X_vals and G_vals:
                        ax.plot(X_vals, G_vals, color=color, linewidth=2, label=f'{ln}³⁺')
                        break
            except:
                continue
        
        ax.set_xlabel('Electric Field (V/Å)')
        ax.set_ylabel('Conductance (S)')
        ax.set_title(f'{ln}³⁺ Stark Effect')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove empty subplot
    if len(lanthanides) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('lanthanide_stark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temperature_dependence():
    """Plot temperature dependence of transport properties."""
    
    # Implementation for temperature plots
    pass

if __name__ == "__main__":
    plot_stark_effect_comparison()
EOF

chmod +x plot_results.py
```

---

## 🎯 **Execution Recommendations**

### **Phase-by-Phase Execution Strategy**

1. **Start with Phase 1** (validation) to establish baseline behavior
2. **Focus on 2-3 lanthanides** initially (Tb, Eu, Er) for different physics
3. **Use `--system_size optimal`** for most studies unless specified otherwise
4. **Run calculations in parallel** where possible using `--parallel_bias`
5. **Save all results** with `--save_json` for later analysis

### **Time Estimates**

| Phase | Calculations | Time per Calc | Total Time |
|-------|-------------|---------------|------------|
| **Phase 1** | ~30 | 2-5 min | 2-3 hours |
| **Phase 2** | ~50 | 3-10 min | 4-8 hours |
| **Phase 3** | ~25 | 5-20 min | 3-8 hours |
| **Phase 4** | ~40 | 2-15 min | 3-10 hours |
| **Phase 5** | ~30 | 5-30 min | 4-15 hours |
| **Phase 6** | ~20 | 2-10 min | 1-3 hours |
| **Total** | ~195 | | **17-47 hours** |

### **Computational Resources**

- **CPU**: Use `--parallel_bias --max_workers 4` for multi-core systems
- **Memory**: 4-8 GB RAM sufficient for most calculations
- **Storage**: ~500 MB for all JSON results and plots
- **Time**: Can be run over several days/weeks as convenient

### **Quality Control**

```bash
# Quick validation check
python graphene_tb_Tb_SW_negf.py --smoke --lanthanide Tb --save_json

# Error checking
python -c "
import glob, json
files = glob.glob('*.json')
for f in files:
    try:
        with open(f) as fp: json.load(fp)
        print(f'✓ {f}')
    except:
        print(f'✗ {f} - CORRUPT')
"
```

This protocol will provide comprehensive characterization of your lanthanide-graphene system with proper physical behavior and statistical validation! 🎉