# Temperature-Dependent Computational Costs

## YES - Different Temperatures Have Different Costs!

Your intuition is correct. The computational cost varies significantly with temperature due to the **adaptive energy grid** system.

## Why This Happens

The energy integration window width depends on temperature:

```
Width = 8.0 × kB × T + 2.0 × kB × T = 10 × kB × T
```

Where kB = 8.617×10⁻⁵ eV/K

### Temperature-to-Grid Mapping

| Temperature Range | Energy Width | Grid Points | Cost Factor |
|-------------------|--------------|-------------|-------------|
| T < 58K           | < 50 meV     | **11 pts**  | 1.0×        |
| 58K < T < 116K    | 50-100 meV   | **21 pts**  | 1.9×        |
| 116K < T < 232K   | 100-200 meV  | **31 pts**  | 2.8×        |
| T > 232K          | > 200 meV    | **41 pts**  | 3.7×        |

## Practical Impact

For a sensitivity sweep with **NX=3** X-points:

| Temperature | Energy Points | NEGF Calls | Relative Cost |
|-------------|---------------|------------|---------------|
| 100K        | 21            | 63         | 1.0×          |
| 200K        | 31            | 93         | 1.5×          |
| 300K        | 41            | 123        | 2.0×          |
| 400K        | 41            | 123        | 2.0×          |

**A 300K calculation takes 2× longer than a 100K calculation!**

## Real-World Example

Consider a 7×5 heat map (35 points):
- **Low-T points** (100K): ~19 seconds each
- **Mid-T points** (200K): ~28 seconds each  
- **High-T points** (300-400K): ~37 seconds each

Total time: ~18.5 minutes, but individual points vary by **2×**!

## Physical Reason

Higher temperatures broaden the Fermi-Dirac distribution, requiring:
1. **Wider energy window** to capture the thermal tails
2. **More sample points** to accurately integrate over the broader distribution
3. **More NEGF calculations** per X-value in sensitivity sweep

At **low T**: Sharp Fermi function → narrow window → few points needed  
At **high T**: Smooth Fermi function → wide window → many points needed

## Optimization Strategies

### 1. Temperature-Aware Batching
Process low-temperature points first (faster feedback):
```bash
# Fast points first
--heatmap_temps 100,150,200,250,300
```

### 2. Coarser High-T Grid (Future Enhancement)
Could reduce high-T cost by accepting smoother integration:
```python
if Temp > 300:
    NE = 31  # Instead of 41
```

### 3. Temperature-Specific Parallelization
If running multiple heat maps, parallelize by temperature ranges:
- Worker 1: Low T (fast, many points)
- Worker 2: High T (slow, fewer points)

### 4. Progress Monitoring
The output shows per-point timing:
```
[  1/  4] T=200.0K, V= 10.0mV → SW: 6.30e-08 S/X  [28.3s]
[  2/  4] T=300.0K, V= 10.0mV → SW: 6.28e-08 S/X  [37.1s]
```

## When to Worry

**Normal variation**: 2× cost difference is expected and acceptable
**Problem indicators**:
- One temperature taking 10× longer → likely different issue (SCF convergence, etc.)
- All points equally slow → GPU not being used
- Random variation → cache misses or memory issues

## Verification

Run the analysis script to see cost breakdown for your specific parameters:
```bash
python temperature_cost_analysis.py
```

This generates:
- Detailed cost tables for different T and V combinations
- Visualization plot showing cost scaling
- Heat map breakdown showing per-point estimates

## Bottom Line

✅ **This is normal behavior** - higher temperatures naturally require more computation  
✅ **Cost scales predictably** - steps at 58K, 116K, and 232K thresholds  
✅ **Can't eliminate entirely** - physics requires wider integration windows at high T  
✅ **Can optimize** - but will always have some T-dependence  

The **2-4× variation** you're seeing is the adaptive grid working as designed, balancing accuracy and speed across the temperature range!
