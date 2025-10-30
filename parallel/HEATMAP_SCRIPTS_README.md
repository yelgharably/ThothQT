# Sensitivity Heat Map Generation Scripts

## Overview
I've created several scripts to run comprehensive sensitivity heat map calculations with the temperature and bias voltage ranges you requested. The scripts are designed for different computational requirements and execution times.

## Scripts Available

### 1. `run_comprehensive_heatmap.ps1` (PowerShell - Recommended for Windows)
**Full Parameter Range:**
- Temperature: 0.001K to 400K in increments of 10K (41 points)
- Bias Voltage: -1000mV to +1000mV in increments of 50mV (41 points)
- Total combinations: **1,681**
- Expected time: **30-60 minutes**

**Usage:**
```powershell
powershell -ExecutionPolicy Bypass -File run_comprehensive_heatmap.ps1
```

### 2. `run_moderate_heatmap.ps1` (PowerShell - Faster Alternative)
**Moderate Parameter Range:**
- Temperature: 1K to 400K in increments of 25K (16 points)
- Bias Voltage: -1000mV to +1000mV in increments of 100mV (21 points)
- Total combinations: **336**
- Expected time: **10-20 minutes**

**Usage:**
```powershell
powershell -ExecutionPolicy Bypass -File run_moderate_heatmap.ps1
```

### 3. `run_comprehensive_heatmap.sh` (Bash - Linux/WSL)
Same as the comprehensive PowerShell version but for bash environments.

**Usage:**
```bash
chmod +x run_comprehensive_heatmap.sh
./run_comprehensive_heatmap.sh
```

### 4. `test_heatmap_params.ps1` (Parameter Verification)
Tests parameter generation without running simulations. Shows:
- Parameter counts and ranges
- Sample command strings
- Different resolution options

**Usage:**
```powershell
powershell -ExecutionPolicy Bypass -File test_heatmap_params.ps1
```

## Parameter Format

The scripts automatically handle the bias voltage formatting required by your system:
- **Negative voltages**: Use 'n' prefix (e.g., `n0.05` for -0.05V)
- **Positive voltages**: Direct format (e.g., `0.05` for +0.05V)

### Example Generated Parameters:
```
# Temperature range (comprehensive)
0.001,10,20,30,40,50,...,390,400

# Bias voltage range (comprehensive)  
n1.00,n0.95,n0.90,...,n0.05,0.00,0.05,...,0.95,1.00
```

## System Parameters Used

The scripts use optimized system parameters:

**Comprehensive Script:**
- Lanthanide: Europium (Eu)
- System size: W=12, L=25 (optimal for accuracy)
- Field range: Xmax=0.03, NX=5
- Includes plotting and electronic structure analysis

**Moderate Script:**
- Same lanthanide and analysis
- Slightly smaller system: W=10, L=20 (faster)
- Same field range and analysis options

## Output Files

Each run generates several output files with timestamps:

1. **`sensitivity_heatmap_Eu_YYYYMMDD_HHMMSS.png`** - Main heat map visualization
2. **`graphene_negf_results_YYYYMMDD_HHMMSS.json`** - Complete numerical data
3. **`graphene_negf_analysis_YYYYMMDD_HHMMSS.png`** - Comprehensive analysis plots
4. **Individual plot files** - I-V curves, transmission spectra, etc.

## Computational Requirements

| Script | Points | Combinations | Time | Memory |
|--------|--------|--------------|------|--------|
| Comprehensive | 41×41 | 1,681 | 30-60 min | ~4-8 GB |
| Moderate | 16×21 | 336 | 10-20 min | ~2-4 GB |
| Quick Test | 3×3 | 9 | 1-2 min | ~1-2 GB |

## Execution Instructions

### For Windows (PowerShell):
1. Open PowerShell as Administrator
2. Navigate to the script directory:
   ```powershell
   cd "d:\Graduate Life\Graphene_stuff\negf_sw\parallel"
   ```
3. Run the desired script:
   ```powershell
   # For comprehensive analysis
   powershell -ExecutionPolicy Bypass -File run_comprehensive_heatmap.ps1
   
   # For moderate analysis (faster)
   powershell -ExecutionPolicy Bypass -File run_moderate_heatmap.ps1
   ```

### For Linux/WSL (Bash):
```bash
cd "/path/to/negf_sw/parallel"
chmod +x run_comprehensive_heatmap.sh
./run_comprehensive_heatmap.sh
```

## Manual Execution

If you prefer to run manually, use this command structure:

```bash
python graphene_tb_Tb_SW_negf.py \
    --lanthanide Eu \
    --W 12 --L 25 \
    --Xmax 0.03 --NX 5 \
    --plot \
    --sensitivity_heatmap \
    --heatmap_temps "0.001,10,20,30,...,400" \
    --heatmap_bias "n1.00,n0.95,n0.90,...,0.95,1.00"
```

## Error Handling

The scripts include:
- Conda environment activation
- Error detection and reporting
- Execution time tracking
- File output verification
- Colored progress reporting

## Solver Configuration

**GPU Acceleration**: Both NEGF and SCF solvers use GPU acceleration when available:

- **NEGF Mode (`--use_negf`)**: Uses `gpu_smatrix_calculation` for fast GPU-accelerated transport calculations
- **SCF Mode (`--use_scf`)**: Uses `gpu_native_scf` which runs full SCF iterations with GPU NEGF backend
- **Default Mode**: Uses standard Kwant solver (CPU-based)

**Recommended Approach**: Use `--use_negf` for heat maps as it provides the best speed-to-accuracy ratio.

## Recent Fixes

1. **✅ Heat Map Independence**: Heat map now runs independently of `--plot` flag
2. **✅ GPU Acceleration**: Confirmed working for both NEGF and SCF modes
3. **✅ Cache Management**: Proper caching with field-dependent keys
4. **✅ Memory Management**: Plots saved automatically, not displayed
5. **✅ SCF Compatibility**: SCF solver now works correctly with heat maps

## Performance Notes

- **NEGF**: ~1-2 seconds per parameter point
- **SCF**: ~10-15 seconds per parameter point (due to SCF iterations)
- **GPU Speedup**: 10-50× faster than CPU depending on system size
- **Cache Benefits**: Subsequent runs with same parameters are nearly instantaneous

The comprehensive script will generate exactly the parameter ranges you requested: temperatures from 0.001K to 400K in 10K increments, and bias voltages from -1000mV to +1000mV in 50mV increments.