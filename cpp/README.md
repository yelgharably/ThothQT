# C++ backend (cpp_negf)

This folder adds a C++/pybind11 backend that accelerates the heavy NEGF kernels, while keeping system construction and orchestration in Python/Kwant.

What’s accelerated (MVP):
- transmission_dense(H, Sigmas, E): dense Tr[Γ_L G^r Γ_R G^a]
- finite_bias_current(H, Sigmas, muL, muR, T_K, E_center, E_span, NE): I(V) via trapezoidal E-integration

Python remains responsible for:
- Building the Kwant system
- Extracting device and lead matrices (H_device, H00/H01, coupling V) and forming self-energies Σ(E)
- SCF loop control and Poisson solver

## Build (Windows, PowerShell)

Prereqs: Python 3.9+, CMake, a C++17 compiler (MSVC Build Tools), and `pip install pybind11 scikit-build-core`.

```
python -m pip install --upgrade pip
python -m pip install pybind11 scikit-build-core
python -m pip wheel . --no-deps -w dist
python -m pip install dist/*.whl
```

Alternatively, to build in-place for development:
```
python -m pip install -e .
```

This produces a `cpp_negf` Python module you can import from `cpp/negf_core.py`.

## Wiring into Python

- In `cpp/negf_core.py`, import `cpp_negf` when available and call it for `transmission`/`finite-bias current` while keeping Python fallbacks.
- In `cpp/scf_solver.py`, use the C++ `finite_bias_current` per SCF iteration when computing current.

## Notes

- This MVP uses a naive dense inversion. For larger devices, replace with Eigen or LAPACK bindings, and parallelize with OpenMP.
- Self-energies Σ must be provided by Python (via existing extraction or an analytic model). A future step could port surface-GF (Lopez-Sancho) to C++ too.
