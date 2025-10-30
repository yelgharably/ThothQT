# Sensitivity Heat Map Documentation

## Overview

The sensitivity heat map functionality allows you to visualize how the field sensitivity (dG/dX) of your graphene NEGF system varies across different temperature and bias voltage conditions. This is useful for:

- Finding optimal operating conditions for maximum sensitivity
- Understanding temperature dependence of device performance  
- Identifying bias voltage ranges with enhanced sensitivity
- Comparing SW defect systems to pristine graphene

## Quick Start

### Method 1: Use the Example Function

```python
from parallel.graphene_tb_Tb_SW_negf import *

# Setup your system
parser = create_argparser()
args = parser.parse_args(['--lanthanide', 'Tb', '--use_scf'])
fsys_sw, fsys_pr = setup_system(args)

# Generate comprehensive heat map analysis
heatmap_data, fig = example_sensitivity_heatmap_usage(fsys_sw, fsys_pr, args)
```

### Method 2: Custom Heat Map

```python
# Define your parameter ranges
temperatures = np.linspace(10, 300, 10)      # 10K to 300K
bias_voltages = np.linspace(-0.1, 0.1, 8)   # -100mV to +100mV

# Compute heat map data
heatmap_data = compute_sensitivity_heatmap(
    fsys_sw=fsys_sw,
    fsys_pr=fsys_pr,
    args=args,
    temperatures=temperatures,
    bias_voltages=bias_voltages
)

# Create plots
fig, ax = plt.subplots(figsize=(10, 8))
plot_sensitivity_heatmap(heatmap_data, args, ax=ax, system_type='SW')
plt.show()
```

### Method 3: Use the Demo Script

```bash
# Quick demo (3x3 parameter grid)
python test_heatmap_demo.py

# Full comprehensive analysis (8x6 parameter grid)  
python test_heatmap_demo.py --full
```

## Functions Reference

### `compute_sensitivity_heatmap(fsys_sw, fsys_pr, args, temperatures, bias_voltages, field_range=None)`

**Purpose**: Computes sensitivity data across temperature-bias voltage parameter space.

**Parameters**:
- `fsys_sw`: SW defect system (kwant.FiniteSystem)
- `fsys_pr`: Pristine system (kwant.FiniteSystem, optional)
- `args`: System parameters (argparse.Namespace)
- `temperatures`: Temperature array [K] (e.g. `np.linspace(10, 300, 8)`)
- `bias_voltages`: Bias voltage array [V] (e.g. `np.linspace(-0.1, 0.1, 6)`)
- `field_range`: Field sweep range [V/Å] (default: `(-args.Xmax, args.Xmax)`)

**Returns**: Dictionary with heat map data including:
- `'sensitivity_sw'`: SW sensitivity values [S per V/Å]
- `'sensitivity_pr'`: Pristine sensitivity (if fsys_pr provided)
- `'sensitivity_enhancement'`: SW/Pristine ratio
- `'conductance_sw'`: Average conductance values
- `'T_grid', 'V_grid'`: Meshgrid coordinates
- `'computation_method'`: 'SCF', 'NEGF', or 'Kwant'

### `plot_sensitivity_heatmap(heatmap_data, args, ax=None, system_type='SW')`

**Purpose**: Creates interpolated heat map visualization.

**Parameters**:
- `heatmap_data`: Output from `compute_sensitivity_heatmap()`
- `args`: System parameters
- `ax`: matplotlib axes (optional, creates new figure if None)  
- `system_type`: 'SW', 'pristine', or 'enhancement'

**Features**:
- Automatic interpolation for smooth heat maps (requires scipy)
- Log scale for sensitivity values
- Contour lines and data point overlays
- Color-coded enhancement factors
- Automatic parameter annotation

### `example_sensitivity_heatmap_usage(fsys_sw, fsys_pr, args)`

**Purpose**: Complete demonstration with multiple heat map types and statistics.

**Features**:
- Default parameter ranges (8 temperatures × 6 bias voltages)
- Multi-panel plots: SW, Pristine, Enhancement, Statistics
- Automatic file saving with timestamps
- Summary statistics printout
- Error handling and fallbacks

## Usage Tips

### 1. Parameter Selection

```python
# For quick testing (fast)
temperatures = np.array([50, 150, 250])           # 3 points
bias_voltages = np.array([-0.05, 0.0, 0.05])     # 3 points

# For detailed analysis (slower)
temperatures = np.linspace(10, 300, 12)           # 12 points  
bias_voltages = np.linspace(-0.15, 0.15, 10)     # 10 points

# For publication-quality (slow)
temperatures = np.linspace(4, 400, 20)            # 20 points
bias_voltages = np.linspace(-0.2, 0.2, 15)       # 15 points
```

### 2. Solver Selection

```python
# For realistic physics (recommended)
args.use_scf = True      # Self-consistent field + NEGF

# For quick testing
args.use_negf = True     # NEGF without self-consistency

# For fastest (but zero-bias only)  
args.use_finite_T = True # Finite temperature Kwant
```

### 3. System Size Considerations

- **Small systems** (length≤20): All solvers work well, fast computation
- **Medium systems** (20<length≤40): SCF/NEGF recommended, moderate speed
- **Large systems** (length>40): GPU acceleration essential, slow computation

### 4. Memory Management

```python
# For large parameter grids, process in chunks
temp_chunks = np.array_split(temperatures, 3)
for i, temp_chunk in enumerate(temp_chunks):
    chunk_data = compute_sensitivity_heatmap(fsys_sw, fsys_pr, args, 
                                           temp_chunk, bias_voltages)
    # Save/process chunk_data before continuing
```

## Expected Results

### Typical Sensitivity Values
- **SW defect systems**: 10⁻⁷ to 10⁻⁵ S per V/Å  
- **Pristine systems**: 10⁻⁸ to 10⁻⁶ S per V/Å
- **Enhancement factors**: 2× to 20× (SW vs pristine)

### Temperature Dependence
- **Low T (10-50K)**: Highest sensitivity, sharp features
- **Room T (250-350K)**: Moderate sensitivity, broader features  
- **High T (>400K)**: Lower sensitivity, thermal broadening

### Bias Voltage Dependence
- **Zero bias**: Often optimal for many systems
- **Finite bias**: Can enhance or reduce sensitivity depending on band structure
- **Large bias**: Generally decreases sensitivity due to broadening

## Troubleshooting

### Common Issues

1. **"Matrix dimension mismatch"**: Usually solved by the automatic padding in negf_core.py
2. **"All zero sensitivity"**: Check Fermi energy, field range, and system parameters
3. **"Memory error"**: Reduce parameter grid size or use chunking
4. **"Slow computation"**: Enable GPU acceleration with `--use_scf` and GPU-enabled system

### Performance Optimization

```python
# Enable all optimizations
args.use_scf = True          # Self-consistent solver
args.use_gpu = True          # GPU acceleration (if available)
args.use_cache = True        # Enable caching
args.NX = 10                 # Reduce field points for testing

# Monitor progress
# Heat map computation shows progress: [23/48] T=150.0K, V=25.0mV → ...
```

## Integration with Existing Code

The heat map functions integrate seamlessly with your existing workflow:

```python
# After running your standard analysis
results_dict = compute_field_sensitivity_analysis(fsys_sw, fsys_pr, args)

# Add heat map analysis  
heatmap_data = compute_sensitivity_heatmap(fsys_sw, fsys_pr, args, 
                                         temperatures, bias_voltages)

# Combine in comprehensive plots
fig = create_sensitivity_parameter_plots(results_dict, args, fsys_sw, fsys_pr)
# Heat map plots can be added to existing figure layout
```

## File Outputs

Heat map analysis automatically generates:
- **PNG files**: `sensitivity_heatmap_{lanthanide}_{timestamp}.png`
- **Console output**: Progress tracking and statistics
- **Return data**: Full numerical results for further analysis

The generated files include professional-quality plots suitable for publications, with proper color scales, contour lines, and parameter annotations.