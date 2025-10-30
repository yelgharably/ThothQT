import numpy as np

try:
    import cpp_negf
    print("cpp_negf import: OK")
except Exception as e:
    print("cpp_negf import failed:", e)
    raise

# Tiny 2x2 toy device, with toy sigmas
H = np.array([[0.1+0j, 0.05],[0.05, -0.1]], dtype=np.complex128)
SigmaL = 1e-3j * np.eye(2, dtype=np.complex128)
SigmaR = 1e-3j * np.eye(2, dtype=np.complex128)

T = np.array(cpp_negf.transmission_dense(H, [SigmaL, SigmaR], 0.0))[0]
print("T(E=0)", T)

I = np.array(cpp_negf.finite_bias_current(H, [SigmaL, SigmaR], 0.005, -0.005, 300.0, 0.0, 0.1, 51))[0]
print("I(V) ~", I)

# Lead blocks for simple symmetric toy lead of size 2x2
H00 = np.array([[0.0+0j, 0.05],[0.05, 0.0]], dtype=np.complex128)
H01 = np.array([[0.0+0j, 0.02],[0.02, 0.0]], dtype=np.complex128)
V   = np.eye(2, dtype=np.complex128) * 0.05

gs = cpp_negf.surface_gf(H00, H01, 0.0, 1e-6)
print("surface_gf(0) shape:", np.array(gs).shape)

Sigma_from_lead = cpp_negf.self_energy_from_lead(H00, H01, V, 0.0, 1e-6)
print("Sigma_from_lead shape:", np.array(Sigma_from_lead).shape)

T2 = cpp_negf.transmission_from_leads(H, H00, H01, V, H00, H01, V, 0.0, 1e-6)
print("T_from_leads(E=0)", float(T2))

Es = np.linspace(-0.1, 0.1, 9)
Ts = cpp_negf.transmission_sweep_from_leads(H, H00, H01, V, H00, H01, V, Es, 1e-6)
print("sweep T size:", np.array(Ts).shape)

I2 = cpp_negf.finite_bias_current_from_leads(H, H00, H01, V, H00, H01, V, 0.005, -0.005, 300.0, 0.0, 0.1, 41, 1e-6)
print("I_from_leads ~", float(I2))
