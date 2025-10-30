#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

// Simple helpers to wrap numpy arrays into Eigen-like operations can be added later.
// For now, use direct pointer math with row-major assumptions or rely on NumPy order.

// Compute transmission using Landauer formula pieces:
// For MVP, accept already-constructed device Green functions or compute a simplified version.
// Here we implement a small dense solver for G^r given H_device and self-energy surrogates.

using cdouble = std::complex<double>;

// ---- Basic dense linear algebra helpers (row-major) ----
static std::vector<cdouble> matmul_rm(const std::vector<cdouble>& A, py::ssize_t r, py::ssize_t k,
                                      const std::vector<cdouble>& B, py::ssize_t k2, py::ssize_t c) {
    if (k != k2) throw std::runtime_error("matmul: inner dim mismatch");
    std::vector<cdouble> C(static_cast<size_t>(r)*static_cast<size_t>(c), cdouble(0,0));
    for (py::ssize_t i=0;i<r;++i) {
        for (py::ssize_t p=0;p<k;++p) {
            cdouble aip = A[static_cast<size_t>(i)*k + p];
            if (aip == cdouble(0,0)) continue;
            for (py::ssize_t j=0;j<c;++j) {
                C[static_cast<size_t>(i)*c + j] += aip * B[static_cast<size_t>(p)*c + j];
            }
        }
    }
    return C;
}

static std::vector<cdouble> conj_transpose_rm(const std::vector<cdouble>& A, py::ssize_t n, py::ssize_t m) {
    std::vector<cdouble> H(static_cast<size_t>(m)*static_cast<size_t>(n));
    for (py::ssize_t i=0;i<n;++i)
        for (py::ssize_t j=0;j<m;++j)
            H[static_cast<size_t>(j)*n + i] = std::conj(A[static_cast<size_t>(i)*m + j]);
    return H;
}

static std::vector<cdouble> identity_rm(py::ssize_t n) {
    std::vector<cdouble> I(static_cast<size_t>(n)*static_cast<size_t>(n), cdouble(0,0));
    for (py::ssize_t i=0;i<n;++i) I[static_cast<size_t>(i)*n + i] = cdouble(1,0);
    return I;
}

static std::vector<cdouble> invert_rm(std::vector<cdouble> A, py::ssize_t n) {
    std::vector<cdouble> Inv = identity_rm(n);
    for (py::ssize_t i=0;i<n;++i) {
        cdouble piv = A[static_cast<size_t>(i)*n + i];
        if (std::abs(piv) == 0.0) throw std::runtime_error("invert_matrix: singular pivot");
        for (py::ssize_t j=0;j<n;++j) {
            A[static_cast<size_t>(i)*n + j] /= piv;
            Inv[static_cast<size_t>(i)*n + j] /= piv;
        }
        for (py::ssize_t k=0;k<n;++k) {
            if (k==i) continue;
            cdouble f = A[static_cast<size_t>(k)*n + i];
            if (f == cdouble(0,0)) continue;
            for (py::ssize_t j=0;j<n;++j) {
                A[static_cast<size_t>(k)*n + j] -= f * A[static_cast<size_t>(i)*n + j];
                Inv[static_cast<size_t>(k)*n + j] -= f * Inv[static_cast<size_t>(i)*n + j];
            }
        }
    }
    return Inv;
}

static std::vector<cdouble> numpy_to_vec_rm(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
    auto a = arr.unchecked<2>();
    py::ssize_t n0 = a.shape(0), n1 = a.shape(1);
    std::vector<cdouble> out(static_cast<size_t>(n0)*static_cast<size_t>(n1));
    for (py::ssize_t i=0;i<n0;++i)
        for (py::ssize_t j=0;j<n1;++j)
            out[static_cast<size_t>(i)*n1 + j] = a(i,j);
    return out;
}

static py::array_t<std::complex<double>> vec_to_numpy_rm(const std::vector<cdouble>& v, py::ssize_t n0, py::ssize_t n1) {
    py::array_t<std::complex<double>> out({n0, n1});
    auto r = out.mutable_unchecked<2>();
    for (py::ssize_t i=0;i<n0;++i)
        for (py::ssize_t j=0;j<n1;++j)
            r(i,j) = v[static_cast<size_t>(i)*n1 + j];
    return out;
}

static py::array_t<double> transmission_dense(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H_device,
    std::vector<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>> Sigmas,
    double energy
) {
    // H_device: (N,N)
    auto H = H_device.unchecked<2>();
    const py::ssize_t N = H.shape(0);
    if (H.shape(1) != N) throw std::runtime_error("H_device must be square");

    // Build E*I - H - sum(Sigma)
    std::vector<cdouble> A(static_cast<size_t>(N)*static_cast<size_t>(N));
    for (py::ssize_t i = 0; i < N; ++i) {
        for (py::ssize_t j = 0; j < N; ++j) {
            A[i*N + j] = -H(i,j);
        }
        A[i*N + i] += cdouble(energy, 1e-6); // add E + i*eta on diagonal
    }

    // Subtract self-energies
    for (auto &Sigma_np : Sigmas) {
        auto S = Sigma_np.unchecked<2>();
        if (S.shape(0) != N || S.shape(1) != N) throw std::runtime_error("Sigma shape mismatch");
        for (py::ssize_t i = 0; i < N; ++i)
            for (py::ssize_t j = 0; j < N; ++j)
                A[i*N + j] -= S(i,j);
    }

    // Solve for G^r = (A)^{-1}
    // Naive Gauss-Jordan inversion (O(N^3)); replace with LAPACK/Eigen for performance if needed.
    std::vector<cdouble> inv(static_cast<size_t>(N)*static_cast<size_t>(N), cdouble(0,0));
    for (py::ssize_t i = 0; i < N; ++i) inv[i*N + i] = cdouble(1,0);

    // Augmented matrix [A | I]
    for (py::ssize_t i = 0; i < N; ++i) {
        // Pivot
        cdouble piv = A[i*N + i];
        if (std::abs(piv) == 0) throw std::runtime_error("Singular matrix in inversion");
        for (py::ssize_t j = 0; j < N; ++j) {
            A[i*N + j] /= piv;
            inv[i*N + j] /= piv;
        }
        // Eliminate
        for (py::ssize_t k = 0; k < N; ++k) {
            if (k == i) continue;
            cdouble f = A[k*N + i];
            if (f == cdouble(0,0)) continue;
            for (py::ssize_t j = 0; j < N; ++j) {
                A[k*N + j] -= f * A[i*N + j];
                inv[k*N + j] -= f * inv[i*N + j];
            }
        }
    }

    // Compute T = Tr[Gamma_L G^r Gamma_R G^a]
    // For the MVP, we expect Sigmas has L and R only, and Gamma = i(Σ - Σ†)
    if (Sigmas.size() < 2) throw std::runtime_error("Need at least two self-energies (left, right)");
    auto SL = Sigmas[0].unchecked<2>();
    auto SR = Sigmas[1].unchecked<2>();

    auto GammaL = std::vector<cdouble>(static_cast<size_t>(N)*static_cast<size_t>(N));
    auto GammaR = std::vector<cdouble>(static_cast<size_t>(N)*static_cast<size_t>(N));
    for (py::ssize_t i = 0; i < N; ++i) {
        for (py::ssize_t j = 0; j < N; ++j) {
            cdouble sL = SL(i,j);
            cdouble sR = SR(i,j);
            GammaL[i*N + j] = cdouble(0,1) * (sL - std::conj(SL(j,i)));
            GammaR[i*N + j] = cdouble(0,1) * (sR - std::conj(SR(j,i)));
        }
    }

    // Compute M = G^r Gamma_R G^a = G^r Gamma_R (G^r)†
    // Then T = Tr[Gamma_L M]
    // Dense multiply: C = A * B
    auto matmul = [&](const std::vector<cdouble>& A_, const std::vector<cdouble>& B_) {
        std::vector<cdouble> C(static_cast<size_t>(N)*static_cast<size_t>(N), cdouble(0,0));
        for (py::ssize_t i = 0; i < N; ++i)
            for (py::ssize_t k = 0; k < N; ++k) {
                cdouble aik = A_[i*N + k];
                if (aik == cdouble(0,0)) continue;
                for (py::ssize_t j = 0; j < N; ++j)
                    C[i*N + j] += aik * B_[k*N + j];
            }
        return C;
    };

    // G^a = (G^r)† -> inv^
    std::vector<cdouble> Gr = inv; // G^r
    std::vector<cdouble> Ga(static_cast<size_t>(N)*static_cast<size_t>(N));
    for (py::ssize_t i = 0; i < N; ++i)
        for (py::ssize_t j = 0; j < N; ++j)
            Ga[i*N + j] = std::conj(Gr[j*N + i]);

    auto Gr_GammaR = matmul(Gr, GammaR);
    auto M = matmul(Gr_GammaR, Ga);

    // T = Tr[GammaL * M]
    double T = 0.0;
    for (py::ssize_t i = 0; i < N; ++i) {
        cdouble sum = 0.0;
        for (py::ssize_t j = 0; j < N; ++j) {
            sum += GammaL[i*N + j] * M[j*N + i];
        }
        T += sum.real();
    }

    auto out = py::array_t<double>(1);
    auto r = out.mutable_unchecked<1>();
    r(0) = T;
    return out;
}

static double fermi(double E, double mu, double kT) {
    if (kT <= 0) return (E <= mu) ? 1.0 : 0.0;
    double x = (E - mu) / kT;
    if (x > 50) return 0.0;
    if (x < -50) return 1.0;
    if (x >= 0) { double ex = std::exp(-x); return ex/(1.0+ex); }
    else { double ex = std::exp(x); return 1.0/(1.0+ex); }
}

static py::array_t<double> finite_bias_current(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H_device,
    std::vector<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>> Sigmas,
    double muL,
    double muR,
    double temperatureK,
    double E_center,
    double E_span,
    int NE
) {
    const double kB_eV_per_K = 8.617333262e-5; // eV/K
    double kT = kB_eV_per_K * temperatureK;

    // Simple trapezoidal integration over energy window centered at E_center
    std::vector<double> Es(NE);
    if (NE < 2) throw std::runtime_error("NE must be >= 2");
    double dE = (E_span) / (NE - 1);
    for (int i = 0; i < NE; ++i) Es[i] = E_center - 0.5*E_span + i * dE;

    double I = 0.0;
    for (int i = 0; i < NE; ++i) {
        // Recompute transmission at each E (slow but clear); reuse transmission_dense core.
        auto T_arr = transmission_dense(H_device, Sigmas, Es[i]);
        double T = T_arr.mutable_unchecked<1>()[0];
        double fL = fermi(Es[i], muL, kT);
        double fR = fermi(Es[i], muR, kT);
        I += T * (fL - fR);
    }
    I *= (2.0 * 1.602176634e-19 / 6.62607015e-34) * dE; // 2e/h * dE (SI)

    auto out = py::array_t<double>(1);
    out.mutable_unchecked<1>()[0] = I;
    return out;
}

// ---- Surface Green's function (Lopez–Sancho) for a semi-infinite lead ----
static py::array_t<std::complex<double>> surface_gf_iterative(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00_np,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01_np,
    double energy,
    double eta,
    int max_iter = 200,
    double tol = 1e-12
) {
    auto H00 = H00_np.unchecked<2>();
    auto H01 = H01_np.unchecked<2>();
    if (H00.shape(0) != H00.shape(1)) throw std::runtime_error("H00 must be square");
    if (H01.shape(0) != H01.shape(1)) throw std::runtime_error("H01 must be square");
    if (H01.shape(0) != H00.shape(0)) throw std::runtime_error("H01 dims must match H00 dims");
    py::ssize_t n = H00.shape(0);

    auto H00v = numpy_to_vec_rm(H00_np);
    auto H01v = numpy_to_vec_rm(H01_np);
    auto H10v = conj_transpose_rm(H01v, n, n);
    auto E = identity_rm(n);
    for (py::ssize_t i=0;i<n;++i) E[static_cast<size_t>(i)*n + i] = cdouble(energy, eta);

    auto alpha = H01v;
    auto beta  = H10v;
    auto eps_s = H00v;

    for (int it=0; it<max_iter; ++it) {
        auto Em_eps = E;
        for (py::ssize_t i=0;i<n;++i)
            for (py::ssize_t j=0;j<n;++j)
                Em_eps[static_cast<size_t>(i)*n + j] -= eps_s[static_cast<size_t>(i)*n + j];

        auto g = invert_rm(Em_eps, n);

        auto alpha_g   = matmul_rm(alpha, n, n, g, n, n);
        auto alpha_new = matmul_rm(alpha_g, n, n, alpha, n, n);
        auto beta_g    = matmul_rm(beta, n, n, g, n, n);
        auto beta_new  = matmul_rm(beta_g, n, n, beta, n, n);

        auto agb = matmul_rm(alpha, n, n, matmul_rm(g, n, n, beta, n, n), n, n);
        auto bga = matmul_rm(beta, n, n, matmul_rm(g, n, n, alpha, n, n), n, n);
        for (py::ssize_t i=0;i<n;++i)
            for (py::ssize_t j=0;j<n;++j)
                eps_s[static_cast<size_t>(i)*n + j] += agb[static_cast<size_t>(i)*n + j] + bga[static_cast<size_t>(i)*n + j];

        double d1 = 0.0, d2 = 0.0;
        for (size_t idx=0; idx<alpha.size(); ++idx) {
            d1 = std::max(d1, std::abs(alpha_new[idx] - alpha[idx]));
            d2 = std::max(d2, std::abs(beta_new[idx]  - beta[idx]));
        }
        alpha.swap(alpha_new);
        beta.swap(beta_new);
        if (std::max(d1, d2) < tol) break;
    }

    auto Em_eps = E;
    for (py::ssize_t i=0;i<n;++i)
        for (py::ssize_t j=0;j<n;++j)
            Em_eps[static_cast<size_t>(i)*n + j] -= eps_s[static_cast<size_t>(i)*n + j];
    auto gs = invert_rm(Em_eps, n);
    return vec_to_numpy_rm(gs, n, n);
}

static py::array_t<std::complex<double>> self_energy_from_lead(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V_couple,
    double energy,
    double eta
) {
    auto gs = surface_gf_iterative(H00, H01, energy, eta);
    auto Vv = numpy_to_vec_rm(V_couple);
    py::ssize_t n_lead = V_couple.unchecked<2>().shape(0);
    py::ssize_t n_dev  = V_couple.unchecked<2>().shape(1);
    auto gsv = numpy_to_vec_rm(gs);

    auto temp = matmul_rm(gsv, n_lead, n_lead, Vv, n_lead, n_dev);
    auto VH   = conj_transpose_rm(Vv, n_lead, n_dev);
    auto Sigma = matmul_rm(VH, n_dev, n_lead, temp, n_lead, n_dev);
    return vec_to_numpy_rm(Sigma, n_dev, n_dev);
}

static double transmission_from_leads(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H_dev,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VL,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VR,
    double energy,
    double eta
) {
    auto Hd = numpy_to_vec_rm(H_dev);
    py::ssize_t nD = H_dev.unchecked<2>().shape(0);
    if (H_dev.unchecked<2>().shape(1) != nD) throw std::runtime_error("H_dev must be square");

    auto SigmaL = numpy_to_vec_rm(self_energy_from_lead(H00L, H01L, VL, energy, eta));
    auto SigmaR = numpy_to_vec_rm(self_energy_from_lead(H00R, H01R, VR, energy, eta));

    std::vector<cdouble> A(static_cast<size_t>(nD)*static_cast<size_t>(nD));
    for (py::ssize_t i=0;i<nD;++i) {
        for (py::ssize_t j=0;j<nD;++j) {
            A[static_cast<size_t>(i)*nD + j] = -Hd[static_cast<size_t>(i)*nD + j] - SigmaL[static_cast<size_t>(i)*nD + j] - SigmaR[static_cast<size_t>(i)*nD + j];
        }
        A[static_cast<size_t>(i)*nD + i] += cdouble(energy, eta);
    }

    auto Gr = invert_rm(A, nD);
    auto Ga = conj_transpose_rm(Gr, nD, nD);

    auto SHL = conj_transpose_rm(SigmaL, nD, nD);
    auto SHR = conj_transpose_rm(SigmaR, nD, nD);
    std::vector<cdouble> GammaL(static_cast<size_t>(nD)*static_cast<size_t>(nD));
    std::vector<cdouble> GammaR(static_cast<size_t>(nD)*static_cast<size_t>(nD));
    for (py::ssize_t i=0;i<nD;++i)
        for (py::ssize_t j=0;j<nD;++j) {
            GammaL[static_cast<size_t>(i)*nD + j] = cdouble(0,1) * (SigmaL[static_cast<size_t>(i)*nD + j] - SHL[static_cast<size_t>(i)*nD + j]);
            GammaR[static_cast<size_t>(i)*nD + j] = cdouble(0,1) * (SigmaR[static_cast<size_t>(i)*nD + j] - SHR[static_cast<size_t>(i)*nD + j]);
        }

    auto Gr_GammaR = matmul_rm(Gr, nD, nD, GammaR, nD, nD);
    auto M = matmul_rm(Gr_GammaR, nD, nD, Ga, nD, nD);
    double T = 0.0;
    for (py::ssize_t i=0;i<nD;++i) {
        cdouble sum = 0.0;
        for (py::ssize_t j=0;j<nD;++j)
            sum += GammaL[static_cast<size_t>(i)*nD + j] * M[static_cast<size_t>(j)*nD + i];
        T += sum.real();
    }
    return T;
}

static py::array_t<double> transmission_sweep_from_leads(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H_dev,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VL,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VR,
    py::array_t<double, py::array::c_style | py::array::forcecast> energies,
    double eta
) {
    auto Es = energies.unchecked<1>();
    py::ssize_t NE = Es.shape(0);
    py::array_t<double> out(NE);
    auto r = out.mutable_unchecked<1>();
    for (py::ssize_t i=0;i<NE;++i) {
        r(i) = transmission_from_leads(H_dev, H00L, H01L, VL, H00R, H01R, VR, Es(i), eta);
    }
    return out;
}

static double finite_bias_current_from_leads(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H_dev,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VL,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H00R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> H01R,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> VR,
    double muL,
    double muR,
    double temperatureK,
    double E_center,
    double E_span,
    int NE,
    double eta
) {
    const double kB_eV_per_K = 8.617333262e-5;
    double kT = kB_eV_per_K * temperatureK;
    if (NE < 2) throw std::runtime_error("NE must be >= 2");
    double dE = (E_span) / (NE - 1);
    auto fermi_loc = [&](double E, double mu){
        if (kT <= 0) return (E <= mu) ? 1.0 : 0.0;
        double x = (E - mu)/kT;
        if (x > 50) return 0.0; if (x < -50) return 1.0;
        if (x >= 0) { double ex = std::exp(-x); return ex/(1.0+ex); }
        else { double ex = std::exp(x); return 1.0/(1.0+ex); }
    };
    double I = 0.0;
    for (int i=0;i<NE;++i) {
        double E = E_center - 0.5*E_span + i * dE;
        double T = transmission_from_leads(H_dev, H00L, H01L, VL, H00R, H01R, VR, E, eta);
        I += T * (fermi_loc(E, muL) - fermi_loc(E, muR));
    }
    I *= (2.0 * 1.602176634e-19 / 6.62607015e-34) * dE;
    return I;
}

PYBIND11_MODULE(cpp_negf, m) {
    m.doc() = "C++ accelerated NEGF kernels (dense MVP)";
    m.def("transmission_dense", &transmission_dense,
          py::arg("H_device"), py::arg("Sigmas"), py::arg("energy"),
          "Compute transmission Tr[Γ_L G^r Γ_R G^a] for dense inputs.");
    m.def("finite_bias_current", &finite_bias_current,
          py::arg("H_device"), py::arg("Sigmas"), py::arg("muL"), py::arg("muR"),
          py::arg("temperatureK"), py::arg("E_center"), py::arg("E_span"), py::arg("NE"),
          "Compute finite-bias current I(V) using trapezoidal integration in energy.");
    m.def("surface_gf", &surface_gf_iterative,
        py::arg("H00"), py::arg("H01"), py::arg("energy"), py::arg("eta"), py::arg("max_iter") = 200, py::arg("tol") = 1e-12,
        "Compute surface Green's function using Lopez–Sancho iteration.");
    m.def("self_energy_from_lead", &self_energy_from_lead,
        py::arg("H00"), py::arg("H01"), py::arg("V_couple"), py::arg("energy"), py::arg("eta"),
        "Compute Σ = V† g_s V from lead blocks.");
    m.def("transmission_from_leads", &transmission_from_leads,
        py::arg("H_device"), py::arg("H00L"), py::arg("H01L"), py::arg("V_L"),
        py::arg("H00R"), py::arg("H01R"), py::arg("V_R"), py::arg("energy"), py::arg("eta"),
        "Compute transmission using H00/H01/V for two leads.");
    m.def("transmission_sweep_from_leads", &transmission_sweep_from_leads,
        py::arg("H_device"), py::arg("H00L"), py::arg("H01L"), py::arg("V_L"),
        py::arg("H00R"), py::arg("H01R"), py::arg("V_R"), py::arg("energies"), py::arg("eta"),
        "Compute T(E) over an array using lead blocks.");
    m.def("finite_bias_current_from_leads", &finite_bias_current_from_leads,
        py::arg("H_device"), py::arg("H00L"), py::arg("H01L"), py::arg("V_L"),
        py::arg("H00R"), py::arg("H01R"), py::arg("V_R"), py::arg("muL"), py::arg("muR"),
        py::arg("temperatureK"), py::arg("E_center"), py::arg("E_span"), py::arg("NE"), py::arg("eta") = 1e-6,
        "Compute finite-bias current using lead blocks.");
}
