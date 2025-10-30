"""
Debug Graphene Zigzag Builder
==============================

Check what's wrong with the transmission.
"""

import numpy as np
import scipy.sparse as sp
from toothqt_production import *
from toothqt_builders import *

print("=" * 80)
print("DEBUG GRAPHENE ZIGZAG RIBBON")
print("=" * 80)
print()

# Build small ribbon for testing
builder = GrapheneBuilder(a=1.42, t=2.7)
device, positions = builder.zigzag_ribbon(width=2, length=3, return_positions=True)

print(f"Device: {device.H.shape[0]} atoms")
print(f"Lead: {device.left.H00.shape[0]} atoms/cell")
print()

# Check device Hamiltonian
print("Device Hamiltonian (should be mostly -t = -2.7):")
H_dense = device.H.toarray()
print(f"Shape: {H_dense.shape}")
print(f"Non-zero elements: {np.count_nonzero(H_dense)}")
print(f"Unique values: {np.unique(np.abs(H_dense[H_dense != 0]))}")
print()

# Check lead Hamiltonians
print("Lead H00:")
print(device.left.H00)
print()

print("Lead H01:")
print(device.left.H01)
print()

# Check self-energy
print("Computing self-energy at E=0.5 eV...")
dec = SanchoRubioDecimator(device.left, eta=1e-4, gpu=False)
g_s = dec.surface_g(0.5)
print(f"Surface g shape: {g_s.shape}")
print(f"g_s diagonal: {np.diag(g_s)}")
print()

Sigma = dec.sigma(0.5)
print(f"Self-energy shape: {Sigma.shape}")
print(f"Sigma diagonal: {np.diag(Sigma)}")
print(f"Im[Sigma] (should be negative): {np.imag(np.diag(Sigma))}")
print()

# Check broadening
Gamma = 1j * (Sigma - Sigma.conj().T)
print(f"Gamma diagonal (should be positive): {np.real(np.diag(Gamma))}")
print()

# Compute transmission
print("Computing transmission...")
engine = NEGFEngine(device, Temp=300, eta=1e-4, gpu=False)
print()

E_test = 0.5
T = engine.transmission(E_test)
print(f"Transmission at E={E_test}: {T:.6e}")
print()

# For comparison, check what the Gamma magnitude is
print("Analysis:")
print(f"  Max |Gamma|: {np.max(np.abs(Gamma)):.6f}")
print(f"  Trace(Gamma_L): {np.trace(Gamma).real:.6f}")
print()

# The issue might be in how the coupling works
# Let's manually check the device-lead coupling
print("Device-Lead Coupling Analysis:")
print("-" * 80)

# The first few sites of device should couple to lead
m = device.left.H00.shape[0]
print(f"Lead has {m} orbitals")
print(f"Device first {m} rows (should connect to lead):")
print(device.H.toarray()[:m, :m])
print()

# Check if there are connections from device to leads
print(f"Device boundary connections:")
print(f"  Top-left block (device[0:{m}, 0:{m}]):")
block_tl = device.H.toarray()[:m, :m]
print(f"    Non-zeros: {np.count_nonzero(block_tl)}")
print(f"    Values: {np.unique(block_tl[block_tl != 0])}")
print()

print(f"  Bottom-right block (device[-{m}:, -{m}:]):")
block_br = device.H.toarray()[-m:, -m:]
print(f"    Non-zeros: {np.count_nonzero(block_br)}")
print(f"    Values: {np.unique(block_br[block_br != 0])}")
print()

# The problem might be that the coupling matrix H01 doesn't match
# how the device is actually structured
print("POTENTIAL ISSUE:")
print("-" * 80)
print("The lead H01 defines how lead cells connect to each other.")
print("But the device boundary needs to connect to the lead in the same way!")
print()
print("Lead H01 says: 'cell N connects to cell N+1 via these hoppings'")
print("Device edge should match this pattern for proper coupling.")
print()

print("=" * 80)
