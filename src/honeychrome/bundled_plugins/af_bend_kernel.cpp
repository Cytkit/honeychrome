// af_bend_kernel.cpp
// ---------------------------------------------------------------------------
// The AF shape bend of per-cell AF (R1) for events already chosen to bend.
// Mirrors controller_components/af_shape_bend.bend_fluorophores (the NumPy
// path), which documents the method; part of the _gls_m2p_kernel extension.
//
// Per event, with leftover d = y - k a_j, per-cell AF fluorophores u, fitted
// signal mu, shapes E (L x D), spectra S (F x D), OLS unmixing matrix P and
// the bend's prior variance T (one value per shape):
//   * detector weights W = 1 / V(mu), V = sigma0^2 + g mu + c mu^2;
//   * the outside term A = E W E' - E W S' (S W S')^-1 S W E' and b likewise
//     with d, recomputed robust_passes times with detectors whose leftover
//     after the bend is beyond robust_z noise SDs down-weighted (Huber);
//   * the fluorophore term g = E P', omega_f = w_f / var_f, var_f the
//     fluorophore's noise variance (V through P^2, plus the spillover
//     spread) and w_f its weight for being near 0;
//   * the bend c = (A + g Omega g' + T^-1)^-1 (b + g Omega u), and each
//     fluorophore corrected by the bend estimated without its own term
//     (rank-one downdate).
//
// Events are processed in chunks: the weighted products of a chunk (S W S'
// through the packed outer products of the spectra, S W E', S W d, E W E',
// E W d) are BLAS dgemm calls, and the per-event work is one F x F and a few
// L x L solves. An event whose weights a robust round leaves unchanged is
// finished without the remaining rounds, which would repeat the same terms
// (n_reweighted counts the events with a down-weighted detector in the first
// round). OpenMP runs chunks in parallel. A system that is not
// positive definite to working precision is solved by elimination with
// partial pivoting; an event whose system is singular keeps its unbent
// fluorophores and is counted in n_failed.
// ---------------------------------------------------------------------------
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel_linalg.hpp"

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

using kernel_linalg::chol_solve;
using kernel_linalg::cholesky;
using kernel_linalg::dot;
using kernel_linalg::gemm_rm;

constexpr int kChunk = 128;
constexpr double kTiny = 1e-300;
constexpr double kFixed = 1e30;        // precision of a bend shape with prior variance 0
constexpr double kVarFloor = 1e-12;    // floor of a fluorophore's noise variance
constexpr double kDownFloor = 1e-12;   // floor of the leave-own-out denominator
constexpr double kInvSqrt2 = 0.70710678118654752440;

// Solve A X = B for a general p x p matrix A (row-major, overwritten) by
// elimination with partial pivoting. B holds nrhs right-hand sides, each
// contiguous (b + c * p), solved in place. False when A is singular.
bool lu_solve(double* a, int p, double* b, int nrhs) {
  for (int j = 0; j < p; ++j) {
    int piv = j;
    double best = std::fabs(a[j * p + j]);
    for (int i = j + 1; i < p; ++i) {
      const double v = std::fabs(a[i * p + j]);
      if (v > best) { best = v; piv = i; }
    }
    if (!(best > 0.0)) return false;
    if (piv != j) {
      for (int k = 0; k < p; ++k) std::swap(a[j * p + k], a[piv * p + k]);
      for (int c = 0; c < nrhs; ++c) std::swap(b[c * p + j], b[c * p + piv]);
    }
    const double inv = 1.0 / a[j * p + j];
    for (int i = j + 1; i < p; ++i) {
      const double f = a[i * p + j] * inv;
      if (f == 0.0) continue;
      for (int k = j + 1; k < p; ++k) a[i * p + k] -= f * a[j * p + k];
      for (int c = 0; c < nrhs; ++c) b[c * p + i] -= f * b[c * p + j];
    }
  }
  for (int c = 0; c < nrhs; ++c) {
    double* x = b + c * p;
    for (int i = p - 1; i >= 0; --i) {
      double s = x[i];
      for (int k = i + 1; k < p; ++k) s -= a[i * p + k] * x[k];
      x[i] = s / a[i * p + i];
    }
  }
  return true;
}

// Solve A X = B for a symmetric p x p matrix A (row-major, both triangles,
// kept), B as in lu_solve; work holds p * p values. Cholesky, or elimination
// with partial pivoting when A is not positive definite. False when singular.
bool sym_solve(const double* a, int p, double* b, int nrhs, double* work) {
  std::memcpy(work, a, sizeof(double) * static_cast<size_t>(p) * p);
  if (cholesky(work, p)) {
    for (int c = 0; c < nrhs; ++c) chol_solve(work, p, b + c * p);
    return true;
  }
  std::memcpy(work, a, sizeof(double) * static_cast<size_t>(p) * p);
  return lu_solve(work, p, b, nrhs);
}

void check(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}

py::dict bend_r1(Arr d_in, Arr u_in, Arr mu_in, Arr t2_in, Arr spectra_in, Arr unmix_in, Arr shapes_in,
                 Arr sigma0_sq_in, Arr g_in, Arr c_in, py::object spread_obj, double robust_z, int robust_passes,
                 double positive_offset_sd, int n_threads) {
  auto db = d_in.request(), ub = u_in.request(), mb = mu_in.request(), tb = t2_in.request();
  auto sb = spectra_in.request(), pb = unmix_in.request(), eb = shapes_in.request();
  auto s0b = sigma0_sq_in.request(), gb = g_in.request(), cb = c_in.request();
  check(db.ndim == 2 && sb.ndim == 2 && eb.ndim == 2, "d, spectra and shapes must be 2-dimensional");
  const int N = static_cast<int>(db.shape[0]), D = static_cast<int>(db.shape[1]);
  const int F = static_cast<int>(sb.shape[0]), L = static_cast<int>(eb.shape[0]);
  check(F > 0 && L > 0 && D > 0, "spectra and shapes must not be empty");
  check(sb.shape[1] == D && eb.shape[1] == D, "spectra must be F x D and shapes L x D");
  check(pb.ndim == 2 && pb.shape[0] == F && pb.shape[1] == D, "unmix_matrix must be F x D");
  check(ub.ndim == 2 && ub.shape[0] == N && ub.shape[1] == F, "u must be N x F");
  check(mb.ndim == 2 && mb.shape[0] == N && mb.shape[1] == D, "mu must be N x D");
  check(tb.ndim == 2 && tb.shape[0] == N && tb.shape[1] == L, "t2 must be N x L");
  check(s0b.size == D && gb.size == D && cb.size == D, "sigma0_sq, g and c must have D values");
  check(robust_passes >= 0, "robust_passes must not be negative");
  const double* dd = static_cast<const double*>(db.ptr);
  const double* uu = static_cast<const double*>(ub.ptr);
  const double* mu = static_cast<const double*>(mb.ptr);
  const double* t2 = static_cast<const double*>(tb.ptr);
  const double* S = static_cast<const double*>(sb.ptr);
  const double* P = static_cast<const double*>(pb.ptr);
  const double* E = static_cast<const double*>(eb.ptr);
  const double* s0 = static_cast<const double*>(s0b.ptr);
  const double* gn = static_cast<const double*>(gb.ptr);
  const double* cn = static_cast<const double*>(cb.ptr);

  std::vector<double> spread;
  if (!spread_obj.is_none()) {
    Arr sp = spread_obj.cast<Arr>();
    auto spb = sp.request();
    check(spb.ndim == 2 && spb.shape[0] == F && spb.shape[1] == F, "spread must be F x F");
    const double* p = static_cast<const double*>(spb.ptr);
    spread.assign(p, p + static_cast<size_t>(F) * F);
  }
  const bool has_spread = !spread.empty();

  // products shared by every event
  const int T = F * (F + 1) / 2, TL = L * (L + 1) / 2, FL = F * L;
  std::vector<double> O(static_cast<size_t>(D) * T), SE(static_cast<size_t>(D) * FL),
      EE(static_cast<size_t>(D) * TL), St(static_cast<size_t>(D) * F), Et(static_cast<size_t>(D) * L),
      P2t(static_cast<size_t>(D) * F), G(static_cast<size_t>(L) * F);
  for (int d = 0; d < D; ++d) {
    int t = 0;
    for (int a = 0; a < F; ++a)
      for (int b = 0; b <= a; ++b, ++t) O[static_cast<size_t>(d) * T + t] = S[a * D + d] * S[b * D + d];
    for (int f = 0; f < F; ++f) {
      for (int l = 0; l < L; ++l) SE[static_cast<size_t>(d) * FL + f * L + l] = S[f * D + d] * E[l * D + d];
      St[d * F + f] = S[f * D + d];
      P2t[d * F + f] = P[f * D + d] * P[f * D + d];
    }
    t = 0;
    for (int l = 0; l < L; ++l) {
      for (int m = 0; m <= l; ++m, ++t) EE[d * TL + t] = E[l * D + d] * E[m * D + d];
      Et[d * L + l] = E[l * D + d];
    }
  }
  for (int l = 0; l < L; ++l)
    for (int f = 0; f < F; ++f) G[l * F + f] = dot(&E[l * D], &P[f * D], D);

  Arr out_np({N, F});
  double* out = static_cast<double*>(out_np.request().ptr);
  const int n_chunks = (N + kChunk - 1) / kChunk;
  long n_failed = 0, n_reweighted = 0;

  {
    py::gil_scoped_release release;
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#pragma omp parallel reduction(+ : n_failed, n_reweighted)
    {
      const size_t Bm = kChunk;
      std::vector<double> W0(Bm * D), W(Bm * D), Wc(Bm * D), WD(Bm * D), VAR(Bm * D), NSS(Bm * T), X(Bm * FL),
          Z(Bm * F), EWE(Bm * TL), EWD(Bm * L), VF(Bm * F), UPOS(has_spread ? Bm * F : 0);
      std::vector<double> M(static_cast<size_t>(F) * F), mwork(static_cast<size_t>(F) * F),
          rhs(static_cast<size_t>(F) * (L + 1)), a(static_cast<size_t>(L) * L), af(static_cast<size_t>(L) * L),
          lwork(static_cast<size_t>(L) * L), b(L), cvec(L), prec(L), refit(F), om(F),
          cols(static_cast<size_t>(L) * (F + 1));
      std::vector<char> ok(Bm);
      std::vector<int> act(Bm);
#pragma omp for schedule(dynamic, 1)
      for (int ch = 0; ch < n_chunks; ++ch) {
        const int i0 = ch * kChunk;
        const int B = std::min(kChunk, N - i0);
        for (int bi = 0; bi < B; ++bi) {
          const double* m = mu + static_cast<size_t>(i0 + bi) * D;
          for (int d = 0; d < D; ++d) {
            const double v = std::max(m[d], 0.0);
            const double var = s0[d] + v * (gn[d] + cn[d] * v);
            VAR[bi * D + d] = var;
            W0[bi * D + d] = W[bi * D + d] = 1.0 / std::max(var, kTiny);
          }
          ok[bi] = 1;
        }
        gemm_rm(B, F, D, VAR.data(), D, P2t.data(), F, VF.data(), F);
        if (has_spread) {
          for (int bi = 0; bi < B; ++bi) {
            const double* u = uu + static_cast<size_t>(i0 + bi) * F;
            for (int f = 0; f < F; ++f) UPOS[bi * F + f] = std::max(u[f], 0.0);
          }
          gemm_rm(B, F, F, UPOS.data(), F, spread.data(), F, VF.data(), F, 1.0);
        }

        // Events still to be refitted (positions in the chunk). An event whose weights a robust
        // round leaves unchanged would get the same terms in every later round, so it is
        // finished at once; the rest are refitted together.
        int n_act = B;
        for (int bi = 0; bi < B; ++bi) act[bi] = bi;
        for (int round = 0; round <= robust_passes && n_act > 0; ++round) {
          for (int ci = 0; ci < n_act; ++ci) {
            const int bi = act[ci];
            const double* dv = dd + static_cast<size_t>(i0 + bi) * D;
            const double* w = &W[bi * D];
            for (int d = 0; d < D; ++d) {
              Wc[ci * D + d] = w[d];
              WD[ci * D + d] = w[d] * dv[d];
            }
          }
          gemm_rm(n_act, T, D, Wc.data(), D, O.data(), T, NSS.data(), T);
          gemm_rm(n_act, FL, D, Wc.data(), D, SE.data(), FL, X.data(), FL);
          gemm_rm(n_act, F, D, WD.data(), D, St.data(), F, Z.data(), F);
          gemm_rm(n_act, TL, D, Wc.data(), D, EE.data(), TL, EWE.data(), TL);
          gemm_rm(n_act, L, D, WD.data(), D, Et.data(), L, EWD.data(), L);

          int n_next = 0;
          for (int ci = 0; ci < n_act; ++ci) {
            const int bi = act[ci];
            const int i = i0 + bi;
            const double* x = &X[static_cast<size_t>(ci) * FL];
            const double* z = &Z[ci * F];
            const double* nss = &NSS[static_cast<size_t>(ci) * T];
            int t = 0;
            for (int f1 = 0; f1 < F; ++f1)
              for (int f2 = 0; f2 <= f1; ++f2, ++t) M[f1 * F + f2] = M[f2 * F + f1] = nss[t];
            // M^-1 [X, z]: columns 0 to L - 1 are M^-1 X, column L is M^-1 z
            for (int l = 0; l < L; ++l)
              for (int f = 0; f < F; ++f) rhs[l * F + f] = x[f * L + l];
            for (int f = 0; f < F; ++f) rhs[L * F + f] = z[f];
            if (!sym_solve(M.data(), F, rhs.data(), L + 1, mwork.data())) { ok[bi] = 0; continue; }
            const double* mz = &rhs[L * F];
            const double* ewe = &EWE[ci * TL];
            t = 0;
            for (int l = 0; l < L; ++l) {
              for (int m = 0; m <= l; ++m, ++t) {
                double s = ewe[t];
                for (int f = 0; f < F; ++f) s -= x[f * L + l] * rhs[m * F + f];
                a[l * L + m] = a[m * L + l] = s;
              }
              double s = EWD[ci * L + l];
              for (int f = 0; f < F; ++f) s -= x[f * L + l] * mz[f];
              b[l] = s;
            }
            const double* tv = t2 + static_cast<size_t>(i) * L;
            for (int l = 0; l < L; ++l) prec[l] = tv[l] > 0.0 ? 1.0 / std::max(tv[l], kTiny) : kFixed;

            if (round < robust_passes) {
              // the bend with these weights, the leftover after it, and the Huber weights
              for (int l = 0; l < L; ++l) {
                for (int m = 0; m < L; ++m) af[l * L + m] = a[l * L + m];
                af[l * L + l] += prec[l];
                cvec[l] = b[l];
              }
              if (!sym_solve(af.data(), L, cvec.data(), 1, lwork.data())) { ok[bi] = 0; continue; }
              for (int f = 0; f < F; ++f) {
                double s = mz[f];
                for (int l = 0; l < L; ++l) s -= rhs[l * F + f] * cvec[l];
                refit[f] = s;
              }
              const double* dv = dd + static_cast<size_t>(i) * D;
              double* w = &W[bi * D];
              const double* w0 = &W0[bi * D];
              bool changed = false;
              for (int d = 0; d < D; ++d) {
                double left = dv[d];
                for (int l = 0; l < L; ++l) left -= cvec[l] * Et[d * L + l];
                left -= dot(refit.data(), &St[d * F], F);
                const double wn =
                    w0[d] * std::min(1.0, robust_z / std::max(std::fabs(left) * std::sqrt(w0[d]), kTiny));
                if (wn != w[d]) changed = true;
                w[d] = wn;
              }
              if (changed) {
                if (round == 0) ++n_reweighted;
                act[n_next++] = bi;
                continue;
              }
            }

            // fluorophore term, the bend, and each fluorophore's leave-own-out correction
            const double* u = uu + static_cast<size_t>(i) * F;
            for (int f = 0; f < F; ++f) {
              const double vf = std::max(VF[bi * F + f], kVarFloor);
              const double sd = std::sqrt(vf);
              om[f] = std::erfc(std::max(u[f] - positive_offset_sd * sd, 0.0) * kInvSqrt2 / sd) / vf;
            }
            for (int l = 0; l < L; ++l) {
              const double* gl = &G[l * F];
              for (int m = 0; m <= l; ++m) {
                const double* gm = &G[m * F];
                double s = a[l * L + m];
                for (int f = 0; f < F; ++f) s += gl[f] * om[f] * gm[f];
                af[l * L + m] = af[m * L + l] = s;
              }
              af[l * L + l] += prec[l];
              double s = b[l];
              for (int f = 0; f < F; ++f) s += om[f] * u[f] * gl[f];
              cols[static_cast<size_t>(F) * L + l] = s;
              for (int f = 0; f < F; ++f) cols[static_cast<size_t>(f) * L + l] = gl[f];
            }
            if (!sym_solve(af.data(), L, cols.data(), F + 1, lwork.data())) { ok[bi] = 0; continue; }
            const double* cb_ = &cols[static_cast<size_t>(F) * L];
            double* o = out + static_cast<size_t>(i) * F;
            for (int f = 0; f < F; ++f) {
              double q = 0.0, cg = 0.0;
              for (int l = 0; l < L; ++l) {
                q += G[l * F + f] * cols[static_cast<size_t>(f) * L + l];
                cg += cb_[l] * G[l * F + f];
              }
              const double delta = (cg - om[f] * u[f] * q) / std::max(1.0 - om[f] * q, kDownFloor);
              o[f] = u[f] - delta;
            }
          }
          n_act = n_next;
        }
        for (int bi = 0; bi < B; ++bi) {
          if (ok[bi]) continue;
          std::memcpy(out + static_cast<size_t>(i0 + bi) * F, uu + static_cast<size_t>(i0 + bi) * F,
                      sizeof(double) * F);
          ++n_failed;
        }
      }
    }
  }

  py::dict result;
  result["unmixed"] = out_np;
  result["n_failed"] = n_failed;
  result["n_reweighted"] = n_reweighted;
  return result;
}

}  // namespace

void register_af_bend(py::module_& m) {
  m.def("bend_r1", &bend_r1, py::arg("d"), py::arg("u"), py::arg("mu"), py::arg("t2"), py::arg("spectra"),
        py::arg("unmix_matrix"), py::arg("shapes"), py::arg("sigma0_sq"), py::arg("g"), py::arg("c"),
        py::arg("spread") = py::none(), py::arg("robust_z") = 4.0, py::arg("robust_passes") = 2,
        py::arg("positive_offset_sd") = 2.0, py::arg("n_threads") = 0);
}
