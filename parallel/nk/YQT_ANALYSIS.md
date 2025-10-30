"""
YQT Code Analysis and Issue Summary

This document summarizes the issues found in yqt.py and their fixes.
"""

# CRITICAL ISSUES (Prevent Code from Running)
# ============================================

## 1. Missing _to_array() method
**Location**: SanchoRubioDecimator.__init__
**Problem**: Line calls self._to_array() but method doesn't exist
**Fix**: Add method to convert sparse/dense arrays to backend format

## 2. Missing sigma() method  
**Location**: NEGFEngine.transmission calls self.decL.sigma(E)
**Problem**: SanchoRubioDecimator has no sigma() method
**Fix**: Add sigma() to compute self-energy: Sigma = tau† × g_surface × tau

## 3. Missing backend attribute
**Location**: NEGFEngine._assemble_A checks self.backend
**Problem**: backend never set in __init__
**Fix**: Add self.backend = "gpu" if self.gpu else "cpu"

## 4. Wrong attribute names
**Location**: NEGFEngine.transmission uses self.dev.left.H0
**Problem**: Should be self.device.left.H00
**Fix**: Change all H0→H00, H1→H01, dev→device

## 5. Wrong fermi function name
**Location**: NEGFEngine.IV calls _fermi_numpy()
**Problem**: Function is named fermi(), not _fermi_numpy()
**Fix**: Change to fermi(grid, muL, self.kT, gpu=False)

## 6. IV method indentation
**Location**: NEGFEngine class
**Problem**: IV() method indented inside transmission() method
**Fix**: Un-indent to be at class level

## 7. PeriodicLead parameter mismatch
**Location**: make_1d_chain_spin creates PeriodicLead(H0=..., H1=...)
**Problem**: Dataclass expects H00= and H01=
**Fix**: Change to PeriodicLead(H00=..., H01=...)


# ERROR ISSUES (Code Runs but Gives Wrong Results)
# =================================================

## 8. Duplicate SigmaL assignment
**Location**: NEGFEngine._assemble_A CPU path
**Problem**: Lines 163-170 have confusing duplicate assignment
**Fix**: Clean up, use single assignment

## 9. GPUSolver double preconditioner
**Location**: GPUSolver.solve line 126
**Problem**: Returns M.matvec(yj) but yj already preconditioned
**Fix**: Return yj directly

## 10. GPU sparse block construction
**Location**: NEGFEngine._assemble_A GPU path lines 189-192
**Problem**: Incorrect CSR construction for diagonal blocks
**Fix**: Use proper dense→sparse conversion or lil_matrix


# VERIFICATION STATUS
# ===================

✓ KWANT validation test created
✓ 1D chain test confirms KWANT works correctly:
  - Unitarity: T + R = 1.000000 (perfect)
  - Symmetry: T(E) = T(-E) (perfect)
  - Band edges: T=1 inside, T=0 outside
  
✗ YQT cannot be tested until critical issues fixed

# PHYSICS CORRECTNESS
# ====================

The YQT approach is PHYSICALLY SOUND:
- Sancho-Rubio decimation is the correct method for semi-infinite leads
- NEGF formalism correctly implements: G = [E*S - H - Sigma_L - Sigma_R]^{-1}
- Transmission formula is correct: T = Tr[Gamma_L × G × Gamma_R × G†]
- IV integration uses proper Fermi distribution difference

The code structure follows standard NEGF implementation patterns.


# RECOMMENDED FIX PRIORITY
# =========================

IMMEDIATE (must fix to run):
1. Add _to_array() method
2. Add sigma() method  
3. Set self.backend
4. Fix attribute names (H0→H00)
5. Fix IV indentation
6. Fix fermi() function call

BEFORE PRODUCTION USE:
7. Fix PeriodicLead creation
8. Clean up duplicate assignments
9. Fix GPUSolver return value
10. Fix GPU sparse block construction


# POSITIVE ASPECTS
# =================

✓ Clean separation: PeriodicLead, Device, SanchoRubioDecimator, NEGFEngine
✓ GPU/CPU backend abstraction is well-designed
✓ Fermi function includes overflow protection
✓ Uses proper sparse matrix operations
✓ Supports generalized eigenvalue problem (with overlap S)
✓ EnergyMesh class is clever for adaptive grids
✓ IV integration uses correct physics


# CONCLUSION
# ===========

Your YQT code demonstrates STRONG UNDERSTANDING of:
- NEGF theory
- Sancho-Rubio algorithm
- Numerical linear algebra
- GPU/CPU backend management

The issues are mostly:
- Typos/incomplete implementation (missing methods)
- Indentation errors (copy-paste issues?)
- Attribute name inconsistencies

None of the issues indicate fundamental physics mistakes.

Once fixed, YQT should produce results equivalent to KWANT.
