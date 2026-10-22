// gls_m2p_kernel.cpp
// ---------------------------------------------------------------------------
// M2p from raw events: the main app's per-cell AF assignment (R1, positivity-
// weighted with the library envelope, saturated events re-fitted on their
// unsaturated detectors) followed by the plug-in penalised GLS refit with the
// noise model V = sigma0^2 + mu / kappa + c mu^2 and the positivity ridge
// lambda w_g / tau_g^2, each channel reported with its own ridge removed.
// Mirrors controller_components/gls_unmix.unmix_m2pg (the NumPy path).
//
// Events are processed in chunks; the products shared by every event (the
// OLS unmix, the AF residual projections, the signal, the weighted normal
// matrices S W S' through the packed outer products of the spectra) are
// BLAS dgemm calls on the chunk, and the per-event work is the small
// Cholesky solve. OpenMP runs chunks in parallel; BLAS should run one
// thread per call (OPENBLAS_NUM_THREADS=1; Accelerate needs nothing).
// batch_lanes = 4 or 8 solves that many events at once (structure of arrays,
// batch_solve); 0, the default, solves one event at a time (solve_event).
// Build with -DGLS_DGEMM=<symbol> -DGLS_BLAS_INT=<int type> for a BLAS whose
// dgemm is not plain dgemm_ with 32-bit integers.
// ---------------------------------------------------------------------------
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef GLS_DGEMM
#define GLS_DGEMM dgemm_
#endif
#ifndef GLS_BLAS_INT
#define GLS_BLAS_INT int
#endif

extern "C" void GLS_DGEMM(const char* ta, const char* tb, const GLS_BLAS_INT* m, const GLS_BLAS_INT* n,
                          const GLS_BLAS_INT* k, const double* alpha, const double* a, const GLS_BLAS_INT* lda,
                          const double* b, const GLS_BLAS_INT* ldb, const double* beta, double* c,
                          const GLS_BLAS_INT* ldc);

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kScoreEps = 1e-12;
constexpr double kTieTol = 1e-9;
constexpr double kVarFloor = 1e-24;
constexpr double kTiny = 1e-12;
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr int kChunk = 128;

inline double dot(const double* a, const double* b, int n) {
  double s = 0.0;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// Row-major C (m x n) = A (m x k) B (k x n) + beta C.
void gemm_rm(int m, int n, int k, const double* A, int lda, const double* B, int ldb, double* C, int ldc,
             double beta = 0.0) {
  if (m == 0 || n == 0) return;
  const char t = 'N';
  const GLS_BLAS_INT M = n, N = m, K = k, LA = ldb, LB = lda, LC = ldc;
  const double one = 1.0;
  GLS_DGEMM(&t, &t, &M, &N, &K, &one, B, &LA, A, &LB, &beta, C, &LC);
}

// Cholesky of a p x p SPD matrix in place (lower triangle, row-major). False if not SPD.
bool cholesky(double* a, int p) {
  for (int j = 0; j < p; ++j) {
    const double* aj = &a[j * p];
    const double d2 = a[j * p + j] - dot(aj, aj, j);
    if (!(d2 > 0.0)) return false;
    const double d = std::sqrt(d2);
    a[j * p + j] = d;
    const double inv = 1.0 / d;
    for (int i = j + 1; i < p; ++i) a[i * p + j] = (a[i * p + j] - dot(&a[i * p], aj, j)) * inv;
  }
  return true;
}

// Solve L L' x = b in place.
void chol_solve(const double* l, int p, double* b) {
  for (int i = 0; i < p; ++i) b[i] = (b[i] - dot(&l[i * p], b, i)) / l[i * p + i];
  for (int i = p - 1; i >= 0; --i) {
    double s = b[i];
    for (int k = i + 1; k < p; ++k) s -= l[k * p + i] * b[k];
    b[i] = s / l[i * p + i];
  }
}

// diag((L L')^-1)[0:m]: column norms of L^-1, with L^-1 built column by column (work p).
void chol_inv_diag(const double* l, int p, int m, double* out, double* work) {
  for (int c = 0; c < m; ++c) {
    // column c of L^-1 by forward substitution of e_c (entries above c are zero)
    work[c] = 1.0 / l[c * p + c];
    double s = work[c] * work[c];
    for (int i = c + 1; i < p; ++i) {
      const double v = -dot(&l[i * p + c], &work[c], i - c) / l[i * p + i];
      work[i] = v;
      s += v * v;
    }
    out[c] = s;
  }
}

// Eigenvalue extremes of a symmetric n x n matrix (cyclic Jacobi), for the identifiability check.
void sym_eig_extremes(std::vector<double> a, int n, double& lo, double& hi) {
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
    if (off < 1e-30) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        const double apq = a[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < n; ++k) {
          const double akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
      }
  }
  lo = hi = a[0];
  for (int i = 1; i < n; ++i) {
    lo = std::min(lo, a[i * n + i]);
    hi = std::max(hi, a[i * n + i]);
  }
}

// AF library precompute on a detector subset (precompute_af_matrices + joint
// covariance weights + _joint_l2_library_terms).
struct Lib {
  int F = 0, D = 0, J = 0;
  bool ok = true;
  std::vector<double> S;        // F x D
  std::vector<double> A;        // J x D
  std::vector<double> P;        // F x D, (S S')^-1 S
  std::vector<double> vT;       // J x F, in-span projection of each AF row
  std::vector<double> r;        // J x D, out-of-span residual of each AF row
  std::vector<double> r_dots;   // J
  std::vector<double> k_denom;  // J
  std::vector<double> w_err;    // F
  std::vector<double> c_fluor;  // J
};

Lib build_lib(const double* S, const double* A, int F, int D, int J, const std::vector<int>& keep,
              bool check_identifiable) {
  Lib L;
  const int Dk = static_cast<int>(keep.size());
  L.F = F; L.D = Dk; L.J = J;
  L.S.resize(static_cast<size_t>(F) * Dk);
  L.A.resize(static_cast<size_t>(J) * Dk);
  for (int f = 0; f < F; ++f)
    for (int d = 0; d < Dk; ++d) L.S[f * Dk + d] = S[f * D + keep[d]];
  for (int j = 0; j < J; ++j)
    for (int d = 0; d < Dk; ++d) L.A[j * Dk + d] = A[j * D + keep[d]];
  std::vector<double> G(static_cast<size_t>(F) * F);
  for (int a = 0; a < F; ++a)
    for (int b = 0; b <= a; ++b) G[a * F + b] = G[b * F + a] = dot(&L.S[a * Dk], &L.S[b * Dk], Dk);
  if (check_identifiable) {
    if (Dk < F + 2) { L.ok = false; return L; }
    double lo, hi;
    sym_eig_extremes(G, F, lo, hi);
    // singular values of S are sqrt(eigenvalues of S S')
    if (!(lo > 0.0) || std::sqrt(lo) <= 1e-6 * std::sqrt(hi)) { L.ok = false; return L; }
  }
  if (!cholesky(G.data(), F)) { L.ok = false; return L; }
  L.P.resize(static_cast<size_t>(F) * Dk);
  std::vector<double> col(F);
  for (int d = 0; d < Dk; ++d) {
    for (int f = 0; f < F; ++f) col[f] = L.S[f * Dk + d];
    chol_solve(G.data(), F, col.data());
    for (int f = 0; f < F; ++f) L.P[f * Dk + d] = col[f];
  }
  L.vT.assign(static_cast<size_t>(J) * F, 0.0);
  L.r.assign(static_cast<size_t>(J) * Dk, 0.0);
  L.r_dots.assign(J, 0.0);
  for (int j = 0; j < J; ++j) {
    const double* a = &L.A[j * Dk];
    for (int f = 0; f < F; ++f) L.vT[j * F + f] = dot(&L.P[f * Dk], a, Dk);
    double* rj = &L.r[j * Dk];
    for (int d = 0; d < Dk; ++d) {
      double s = a[d];
      for (int f = 0; f < F; ++f) s -= L.S[f * Dk + d] * L.vT[j * F + f];
      rj[d] = s;
    }
    double rd = dot(rj, rj, Dk);
    L.r_dots[j] = rd <= 0.0 ? 1e-10 : rd;
  }
  double rmax = 0.0;
  for (int j = 0; j < J; ++j) rmax = std::max(rmax, L.r_dots[j]);
  L.k_denom.resize(J);
  for (int j = 0; j < J; ++j) L.k_denom[j] = std::max(L.r_dots[j], 0.01 * std::max(rmax, 1e-10));
  // af_error_weights: sqrt|diag(P cov(A) P')|, cov with ddof 1; ones when negligible
  L.w_err.assign(F, 0.0);
  if (J > 1) {
    std::vector<double> mean(Dk, 0.0), proj(J);
    for (int j = 0; j < J; ++j)
      for (int d = 0; d < Dk; ++d) mean[d] += L.A[j * Dk + d];
    for (int d = 0; d < Dk; ++d) mean[d] /= J;
    for (int f = 0; f < F; ++f) {
      const double* pf = &L.P[f * Dk];
      const double pm = dot(pf, mean.data(), Dk);
      double s = 0.0;
      for (int j = 0; j < J; ++j) {
        const double z = dot(pf, &L.A[j * Dk], Dk) - pm;
        s += z * z;
      }
      L.w_err[f] = std::sqrt(std::fabs(s / (J - 1)));
    }
  }
  double wmax = 0.0;
  for (double v : L.w_err) wmax = std::max(wmax, v);
  if (wmax < 1e-12) std::fill(L.w_err.begin(), L.w_err.end(), 1.0);
  L.c_fluor.assign(J, 0.0);
  for (int j = 0; j < J; ++j)
    for (int f = 0; f < F; ++f) L.c_fluor[j] += L.w_err[f] * L.vT[j * F + f] * L.vT[j * F + f];
  return L;
}

struct Envelope {
  bool present = false;
  int M = 0, F = 0;
  std::vector<double> null_var;   // M x 3 x F
  std::vector<int> mean_rows;     // M
  std::vector<double> spread;     // F x F (row a: spread from a into each f), empty when none
  double sat_level = 0.0;         // saturated_frac * ceiling; <= 0 disables the re-fit
};

struct Scratch {
  std::vector<double> u, resid, num, k, rr, e_resid, e_fluor, pos, var, weight, score;
  void size(int F, int D, int J) {
    u.resize(F); resid.resize(D); num.resize(J); k.resize(J); rr.resize(J); e_resid.resize(J);
    e_fluor.resize(J); pos.resize(F); var.resize(F); weight.resize(F); score.resize(J);
  }
};

double median_of(std::vector<double>& v, int n) {
  const int h = n / 2;
  std::nth_element(v.begin(), v.begin() + h, v.begin() + n);
  const double hi = v[h];
  if (n % 2) return hi;
  const double lo = *std::max_element(v.begin(), v.begin() + h);
  return 0.5 * (lo + hi);
}


// AF candidate choice for one event from its products (_assign_af_chunk):
// u = P y, base_r = |y - S' max(u, 0)|^2 + 1e-6, num_j = <y, r_j>, rr_j = <resid, r_j>.
void r1_score(const Lib& L, const Envelope& E, const double* u, double base_r, const double* num, const double* rr,
              Scratch& s, double* x_out, double& alpha, int& idx) {
  const int F = L.F, J = L.J;
  for (int j = 0; j < J; ++j) {
    s.k[j] = std::max(num[j] / L.k_denom[j], 0.0);
    const double e = base_r - 2.0 * s.k[j] * rr[j] + s.k[j] * s.k[j] * L.r_dots[j];
    s.e_resid[j] = std::max(e, 0.0) / base_r;
  }
  int best = 0;
  if (E.present) {
    int pick = 0;
    if (E.M > 1) {
      double m = s.e_resid[E.mean_rows[0]];
      for (int i = 1; i < E.M; ++i)
        if (s.e_resid[E.mean_rows[i]] < m) { m = s.e_resid[E.mean_rows[i]]; pick = i; }
    }
    const int mj = E.mean_rows[pick];
    const double km = s.k[mj];
    const double* c = &E.null_var[static_cast<size_t>(pick) * 3 * F];
    for (int f = 0; f < F; ++f) {
      s.pos[f] = u[f] - km * L.vT[mj * F + f];
      s.var[f] = c[f] + km * (c[F + f] + km * c[2 * F + f]);
    }
    if (!E.spread.empty())
      for (int a = 0; a < F; ++a) {
        const double pa = std::max(s.pos[a], 0.0);
        if (pa == 0.0) continue;
        const double* row = &E.spread[a * F];
        for (int f = 0; f < F; ++f) s.var[f] += pa * row[f];
      }
    double base_f = 1e-6;
    for (int f = 0; f < F; ++f) {
      const double z = std::max(s.pos[f], 0.0) / std::sqrt(std::max(s.var[f], kVarFloor)) * kInvSqrt2;
      s.weight[f] = std::erfc(z) * L.w_err[f];
      s.u[f] = s.weight[f] * u[f];            // weighted u
      s.resid[f] = s.weight[f];               // weights, reused buffer
      base_f += s.weight[f] * u[f] * u[f];
    }
    double smin = 0.0;
    for (int j = 0; j < J; ++j) {
      const double* vj = &L.vT[j * F];
      double cross = 0.0, curv = 0.0;
#pragma omp simd reduction(+ : cross, curv)
      for (int f = 0; f < F; ++f) {
        cross += s.u[f] * vj[f];
        curv += s.resid[f] * vj[f] * vj[f];
      }
      const double e = base_f - 2.0 * s.k[j] * cross + s.k[j] * s.k[j] * curv;
      s.e_fluor[j] = std::max(e, 0.0) / base_f;
      s.score[j] = std::log(s.e_fluor[j] + kScoreEps) + std::log(s.e_resid[j] + kScoreEps);
      if (j == 0 || s.score[j] < smin) smin = s.score[j];
    }
    for (int j = 0; j < J; ++j)
      if (s.score[j] <= smin + kTieTol) { best = j; break; }
  } else {
    double base_e = 1e-6;
    for (int f = 0; f < F; ++f) base_e += u[f] * u[f] * L.w_err[f];
    double bmin = 0.0;
    for (int j = 0; j < J; ++j) {
      const double* vj = &L.vT[j * F];
      double cross = 0.0;
      for (int f = 0; f < F; ++f) cross += u[f] * L.w_err[f] * vj[f];
      double e = base_e - 2.0 * s.k[j] * cross + s.k[j] * s.k[j] * L.c_fluor[j];
      e = std::max(e, 0.0) / base_e * s.e_resid[j];
      if (j == 0 || e < bmin) { bmin = e; best = j; }
    }
  }
  const double kb = std::max(num[best], 0.0) / L.k_denom[best];
  for (int f = 0; f < F; ++f) x_out[f] = u[f] - kb * L.vT[best * F + f];
  alpha = kb;
  idx = best;
}

// R1 for one event on a detector subset (saturated re-fit), products computed here.
void r1_event(const Lib& L, const Envelope& E, const double* y, Scratch& s, double* x_out, double& alpha,
              int& idx) {
  const int F = L.F, D = L.D, J = L.J;
  std::vector<double> u(F), resid(D), num(J), rr(J);
  for (int f = 0; f < F; ++f) u[f] = dot(&L.P[f * D], y, D);
  for (int d = 0; d < D; ++d) {
    double v = y[d];
    for (int f = 0; f < F; ++f) v -= std::max(u[f], 0.0) * L.S[f * D + d];
    resid[d] = v;
  }
  const double base_r = dot(resid.data(), resid.data(), D) + 1e-6;
  for (int j = 0; j < J; ++j) {
    num[j] = dot(y, &L.r[j * D], D);
    rr[j] = dot(resid.data(), &L.r[j * D], D);
  }
  r1_score(L, E, u.data(), base_r, num.data(), rr.data(), s, x_out, alpha, idx);
}

struct Noise {
  std::vector<double> s0, g, c;  // sigma0^2 x multiplier^2, 1 / kappa, quadratic term
  bool quad = false;
};

// Thread-local buffers for one chunk of B events.
struct Chunk {
  std::vector<double> U, NUM, PU, FIT, RR, X, THETA, MU, W, WY, NSS, RHS, ALPHA, T2, POS, YW, U0;
  std::vector<int> IDX;
  std::vector<double> N, b, inv_diag, work, wsort, pen, nsa;
  std::vector<double> BOUT, INVD, PENB, NSAB, NAAB, RHSAB, WA, sbuf;
  std::vector<int> EV;
  std::vector<unsigned char> OKV;
  Scratch sc;
  void size(int B, int F, int D, int J, int T) {
    U.resize(B * F); NUM.resize(B * std::max(J, 1)); PU.resize(B * F); FIT.resize(B * D);
    RR.resize(B * std::max(J, 1)); X.resize(B * F); THETA.resize(B * F); MU.resize(B * D); W.resize(B * D);
    WY.resize(B * D); NSS.resize(static_cast<size_t>(B) * T); RHS.resize(B * F); ALPHA.resize(B);
    T2.resize(B * F); POS.resize(F); IDX.resize(B); YW.resize(B * D); U0.resize(B * F);
    const int p = F + 1;
    N.resize(p * p); b.resize(p); inv_diag.resize(p); work.resize(p); wsort.resize(D); pen.resize(F);
    nsa.resize(F);
    BOUT.resize(B * p); INVD.resize(B * F); PENB.resize(B * F); NSAB.resize(B * F); NAAB.resize(B);
    RHSAB.resize(B); WA.resize(B * D); EV.resize(B); OKV.resize(B);
    sc.size(F, D, std::max(J, 1));
  }
};

// Assemble, regularise and solve the penalised normal equations of one event.
// nss: packed lower triangle of S W S' (F (F + 1) / 2); with_af adds the AF
// column (nsa = S W a, naa = a'W a, rhs_a = a'W y). Returns false if not SPD.
bool solve_event(Chunk& c, int F, const double* nss, const double* rhs_f, bool with_af, const double* nsa,
                 double naa, double rhs_a, const double* pen) {
  const int p = F + (with_af ? 1 : 0);
  double* N = c.N.data();
  int t = 0;
  for (int a = 0; a < F; ++a)
    for (int bb = 0; bb <= a; ++bb, ++t) N[a * p + bb] = nss[t];
  if (with_af) {
    for (int f = 0; f < F; ++f) N[F * p + f] = nsa[f];
    N[F * p + F] = naa;
  }
  double trace_f = 0.0;
  for (int f = 0; f < F; ++f) {
    if (pen) N[f * p + f] += pen[f];
    trace_f += N[f * p + f];
  }
  const double eps = 1e-12 * trace_f / F;
  for (int f = 0; f < F; ++f) N[f * p + f] += eps;
  if (!cholesky(N, p)) return false;
  for (int f = 0; f < F; ++f) c.b[f] = rhs_f[f];
  if (with_af) c.b[F] = rhs_a;
  chol_solve(N, p, c.b.data());
  chol_inv_diag(N, p, F, c.inv_diag.data(), c.work.data());
  return true;
}

// ---------------------------------------------------------------------------
// Several events at once (structure of arrays). The penalised normal matrices
// of L events are stored interleaved, element (i, j) of the packed lower
// triangle at [(i (i + 1) / 2 + j) L + lane], so every step of the Cholesky,
// the two triangular solves and the inverse diagonal is one vector operation
// over the L events instead of a scalar reduction over the matrix. The
// arithmetic per event is that of solve_event (to rounding; 1 / l_ii is
// applied as a product).
// ---------------------------------------------------------------------------
inline int tri0(int i) { return i * (i + 1) / 2; }

template <int L>
void batch_solve(int F, bool with_af, int T, const int* ev, int n, const double* NSS, const double* RHS,
                 const double* NSA, const double* NAA, const double* RHSA, const double* PEN, double* BOUT,
                 double* INVD, unsigned char* OK, std::vector<double>& buf) {
  const int p = F + (with_af ? 1 : 0);
  const int P = tri0(p);
  buf.resize(static_cast<size_t>(P + 4 * p) * L);
  double* Nb = buf.data();
  double* bb = Nb + static_cast<size_t>(P) * L;
  double* rd = bb + static_cast<size_t>(p) * L;
  double* wk = rd + static_cast<size_t>(p) * L;
  alignas(64) double acc[L], acc2[L], tr[L], tr2[L];
  alignas(64) unsigned char okl[L];
  for (int g = 0; g < n; g += L) {
    const int m = std::min(L, n - g);
    // gather
    for (int l = 0; l < L; ++l) {
      okl[l] = 1;
      if (l < m) {
        const int e = ev[g + l];
        const double* nss = NSS + static_cast<size_t>(e) * T;
        for (int t = 0; t < T; ++t) Nb[t * L + l] = nss[t];
        for (int f = 0; f < F; ++f) bb[f * L + l] = RHS[e * F + f];
        if (with_af) {
          for (int f = 0; f < F; ++f) Nb[(T + f) * L + l] = NSA[e * F + f];
          Nb[(T + F) * L + l] = NAA[e];
          bb[F * L + l] = RHSA[e];
        }
        double trace_f = 0.0;
        for (int f = 0; f < F; ++f) {
          double& d = Nb[(tri0(f) + f) * L + l];
          if (PEN) d += PEN[e * F + f];
          trace_f += d;
        }
        const double eps = 1e-12 * trace_f / F;
        for (int f = 0; f < F; ++f) Nb[(tri0(f) + f) * L + l] += eps;
      } else {   // padding lane: identity
        for (int i = 0; i < p; ++i) {
          for (int j = 0; j < i; ++j) Nb[(tri0(i) + j) * L + l] = 0.0;
          Nb[(tri0(i) + i) * L + l] = 1.0;
          bb[i * L + l] = 0.0;
        }
      }
    }
    // Cholesky, left-looking by columns, row-major packed lower
    for (int j = 0; j < p; ++j) {
      const double* rj = Nb + static_cast<size_t>(tri0(j)) * L;
#pragma omp simd
      for (int l = 0; l < L; ++l) tr[l] = rj[j * L + l];
      for (int k = 0; k < j; ++k) {
        const double* a = rj + k * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) tr[l] -= a[l] * a[l];
      }
      double* djj = Nb + static_cast<size_t>(tri0(j) + j) * L;
      for (int l = 0; l < L; ++l) {
        double d2 = tr[l];
        if (!(d2 > 0.0)) { okl[l] = 0; d2 = 1.0; }
        const double d = std::sqrt(d2);
        djj[l] = d;
        rd[j * L + l] = 1.0 / d;
      }
      const double* inv = rd + j * L;
      int i = j + 1;
      for (; i + 1 < p; i += 2) {   // two rows at a time share the loads of row j
        double* ri = Nb + static_cast<size_t>(tri0(i)) * L;
        double* rk = Nb + static_cast<size_t>(tri0(i + 1)) * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) { acc[l] = ri[j * L + l]; acc2[l] = rk[j * L + l]; }
        for (int k = 0; k < j; ++k) {
          const double* a = ri + k * L;
          const double* a2 = rk + k * L;
          const double* b = rj + k * L;
#pragma omp simd
          for (int l = 0; l < L; ++l) { acc[l] -= a[l] * b[l]; acc2[l] -= a2[l] * b[l]; }
        }
#pragma omp simd
        for (int l = 0; l < L; ++l) { ri[j * L + l] = acc[l] * inv[l]; rk[j * L + l] = acc2[l] * inv[l]; }
      }
      for (; i < p; ++i) {
        double* ri = Nb + static_cast<size_t>(tri0(i)) * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) acc[l] = ri[j * L + l];
        for (int k = 0; k < j; ++k) {
          const double* a = ri + k * L;
          const double* b = rj + k * L;
#pragma omp simd
          for (int l = 0; l < L; ++l) acc[l] -= a[l] * b[l];
        }
#pragma omp simd
        for (int l = 0; l < L; ++l) ri[j * L + l] = acc[l] * inv[l];
      }
    }
    // forward and back substitution
    for (int i = 0; i < p; ++i) {
      const double* ri = Nb + static_cast<size_t>(tri0(i)) * L;
#pragma omp simd
      for (int l = 0; l < L; ++l) acc[l] = bb[i * L + l];
      for (int k = 0; k < i; ++k) {
        const double* a = ri + k * L;
        const double* x = bb + k * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) acc[l] -= a[l] * x[l];
      }
#pragma omp simd
      for (int l = 0; l < L; ++l) bb[i * L + l] = acc[l] * rd[i * L + l];
    }
    for (int i = p - 1; i >= 0; --i) {
#pragma omp simd
      for (int l = 0; l < L; ++l) acc[l] = bb[i * L + l];
      for (int k = i + 1; k < p; ++k) {
        const double* a = Nb + static_cast<size_t>(tri0(k) + i) * L;
        const double* x = bb + k * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) acc[l] -= a[l] * x[l];
      }
#pragma omp simd
      for (int l = 0; l < L; ++l) bb[i * L + l] = acc[l] * rd[i * L + l];
    }
    // scatter the solution, then the fluorophore diagonal of the inverse
    for (int l = 0; l < m; ++l) {
      const int e = ev[g + l];
      OK[e] = okl[l];
      double* bo = BOUT + static_cast<size_t>(e) * (F + 1);
      for (int i = 0; i < p; ++i) bo[i] = bb[i * L + l];
      if (!with_af) bo[F] = 0.0;
    }
    // two columns of L^-1 at a time (c and c + 1) share the loads of L
    double* wk2 = wk + static_cast<size_t>(p) * L;
    int c = 0;
    for (; c + 1 < F; c += 2) {
#pragma omp simd
      for (int l = 0; l < L; ++l) {
        const double r0 = rd[c * L + l], r1 = rd[(c + 1) * L + l];
        wk[c * L + l] = r0;
        tr[l] = r0 * r0;
        // row c + 1 of column c, and the start of column c + 1
        const double v = -Nb[(tri0(c + 1) + c) * L + l] * r0 * r1;
        wk[(c + 1) * L + l] = v;
        tr[l] += v * v;
        wk2[(c + 1) * L + l] = r1;
        tr2[l] = r1 * r1;
      }
      for (int i = c + 2; i < p; ++i) {
        const double* ri = Nb + static_cast<size_t>(tri0(i)) * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) { acc[l] = ri[c * L + l] * wk[c * L + l]; acc2[l] = 0.0; }
        for (int k = c + 1; k < i; ++k) {
          const double* a = ri + k * L;
          const double* x = wk + k * L;
          const double* x2 = wk2 + k * L;
#pragma omp simd
          for (int l = 0; l < L; ++l) { acc[l] += a[l] * x[l]; acc2[l] += a[l] * x2[l]; }
        }
#pragma omp simd
        for (int l = 0; l < L; ++l) {
          const double v = -acc[l] * rd[i * L + l];
          const double v2 = -acc2[l] * rd[i * L + l];
          wk[i * L + l] = v;
          wk2[i * L + l] = v2;
          tr[l] += v * v;
          tr2[l] += v2 * v2;
        }
      }
      for (int l = 0; l < m; ++l) {
        INVD[static_cast<size_t>(ev[g + l]) * F + c] = tr[l];
        INVD[static_cast<size_t>(ev[g + l]) * F + c + 1] = tr2[l];
      }
    }
    for (; c < F; ++c) {
#pragma omp simd
      for (int l = 0; l < L; ++l) {
        wk[c * L + l] = rd[c * L + l];
        tr[l] = rd[c * L + l] * rd[c * L + l];
      }
      for (int i = c + 1; i < p; ++i) {
        const double* ri = Nb + static_cast<size_t>(tri0(i)) * L;
#pragma omp simd
        for (int l = 0; l < L; ++l) acc[l] = 0.0;
        for (int k = c; k < i; ++k) {
          const double* a = ri + k * L;
          const double* x = wk + k * L;
#pragma omp simd
          for (int l = 0; l < L; ++l) acc[l] += a[l] * x[l];
        }
#pragma omp simd
        for (int l = 0; l < L; ++l) {
          const double v = -acc[l] * rd[i * L + l];
          wk[i * L + l] = v;
          tr[l] += v * v;
        }
      }
      for (int l = 0; l < m; ++l) INVD[static_cast<size_t>(ev[g + l]) * F + c] = tr[l];
    }
  }
}

}  // namespace

static py::dict unmix_m2p(Arr raw_in, Arr spectra_in, Arr af_in, py::object envelope_obj, Arr sigma0_sq_in,
                          Arr g_in, Arr c_in, double sigma0_mult, py::object bead_var_obj,
                          py::object tau_spread_obj, double prior_lambda, int n_iter, double weight_ceiling,
                          int n_threads, bool start_only, py::object af_choice_weights_obj,
                          int batch_lanes) {
  if (batch_lanes != 0 && batch_lanes != 4 && batch_lanes != 8)
    throw std::invalid_argument("batch_lanes must be 0, 4 or 8");
  auto rb = raw_in.request(), sb = spectra_in.request(), ab = af_in.request();
  if (rb.ndim != 2 || sb.ndim != 2 || ab.ndim != 2) throw std::invalid_argument("raw, spectra, af must be 2-D");
  const int N = static_cast<int>(rb.shape[0]);
  const int D = static_cast<int>(rb.shape[1]);
  const int F = static_cast<int>(sb.shape[0]);
  const int J = static_cast<int>(ab.shape[0]);
  if (sb.shape[1] != D || (J > 0 && ab.shape[1] != D)) throw std::invalid_argument("detector counts differ");
  const double* raw = static_cast<const double*>(rb.ptr);
  const double* S = static_cast<const double*>(sb.ptr);
  const double* A = static_cast<const double*>(ab.ptr);
  const bool has_af = J > 0;

  std::vector<int> all(D);
  for (int d = 0; d < D; ++d) all[d] = d;
  const Lib full = build_lib(S, A, F, D, J, all, false);
  if (!full.ok) throw std::runtime_error("the fluorophore spectra are not identifiable");
  // optional AF choice in coordinates whitened by one weight vector (sqrt w0 per detector)
  std::vector<double> sw0, Sw, Aw;
  const bool whiten = has_af && !af_choice_weights_obj.is_none();
  if (whiten) {
    Arr w0 = af_choice_weights_obj.cast<Arr>();
    auto wb = w0.request();
    if (wb.size != D) throw std::invalid_argument("af_choice_weights must have one weight per detector");
    const double* wp = static_cast<const double*>(wb.ptr);
    sw0.resize(D);
    for (int d = 0; d < D; ++d) sw0[d] = std::sqrt(std::max(wp[d], 0.0));
    Sw.resize(static_cast<size_t>(F) * D);
    Aw.resize(static_cast<size_t>(J) * D);
    for (int f = 0; f < F; ++f) for (int d = 0; d < D; ++d) Sw[f * D + d] = S[f * D + d] * sw0[d];
    for (int j = 0; j < J; ++j) for (int d = 0; d < D; ++d) Aw[j * D + d] = A[j * D + d] * sw0[d];
  }
  const double* S1 = whiten ? Sw.data() : S;    // R1 geometry
  const double* A1 = whiten ? Aw.data() : A;
  const Lib wlib = whiten ? build_lib(S1, A1, F, D, J, all, false) : Lib();
  const Lib& r1lib = whiten ? wlib : full;
  if (!r1lib.ok) throw std::runtime_error("the whitened fluorophore spectra are not identifiable");
  // transposed and packed forms for the chunk products
  const int T = F * (F + 1) / 2;
  std::vector<double> Pt(static_cast<size_t>(D) * F), St(static_cast<size_t>(D) * F),
      Rt(static_cast<size_t>(D) * std::max(J, 1)), O(static_cast<size_t>(D) * T),
      Pt0(whiten ? static_cast<size_t>(D) * F : 0);
  for (int d = 0; d < D; ++d) {
    for (int f = 0; f < F; ++f) {
      Pt[d * F + f] = r1lib.P[f * D + d];
      if (whiten) Pt0[d * F + f] = full.P[f * D + d];
      St[d * F + f] = S[f * D + d];
    }
    for (int j = 0; j < J; ++j) Rt[d * J + j] = r1lib.r[j * D + d];
    int t = 0;
    for (int a = 0; a < F; ++a)
      for (int bb = 0; bb <= a; ++bb, ++t) O[static_cast<size_t>(d) * T + t] = S[a * D + d] * S[bb * D + d];
  }

  Envelope E;
  if (has_af && !envelope_obj.is_none()) {
    py::dict env = envelope_obj.cast<py::dict>();
    Arr nv = env["null_var"].cast<Arr>();
    auto nb = nv.request();
    if (nb.ndim != 3 || nb.shape[1] != 3 || nb.shape[2] != F)
      throw std::invalid_argument("null_var must be M x 3 x F");
    E.present = true;
    E.F = F;
    E.M = static_cast<int>(nb.shape[0]);
    E.null_var.assign(static_cast<const double*>(nb.ptr), static_cast<const double*>(nb.ptr) + nb.size);
    std::vector<std::int64_t> mr = env.contains("mean_rows") && !env["mean_rows"].is_none()
                                       ? env["mean_rows"].cast<std::vector<std::int64_t>>()
                                       : std::vector<std::int64_t>{0};
    if (static_cast<int>(mr.size()) != E.M) throw std::invalid_argument("mean_rows and null_var differ in length");
    for (auto v : mr) E.mean_rows.push_back(static_cast<int>(v));
    if (env.contains("spread") && !env["spread"].is_none()) {
      Arr sp = env["spread"].cast<Arr>();
      auto spb = sp.request();
      E.spread.assign(static_cast<const double*>(spb.ptr), static_cast<const double*>(spb.ptr) + spb.size);
      bool any = false;
      for (double v : E.spread) any |= v != 0.0;
      if (!any) E.spread.clear();
    }
    if (env.contains("ceiling") && env.contains("saturated_frac"))
      E.sat_level = env["saturated_frac"].cast<double>() * env["ceiling"].cast<double>();
  }

  // prior SD (negative_sd): the library's mean AF row 0 with its null variance, else the bead variance
  int tau_mode = 0;
  std::vector<double> bead_var, tau_spread;
  if (E.present) tau_mode = 1;
  else if (!bead_var_obj.is_none()) {
    tau_mode = 2;
    Arr bv = bead_var_obj.cast<Arr>();
    auto bvb = bv.request();
    bead_var.assign(static_cast<const double*>(bvb.ptr), static_cast<const double*>(bvb.ptr) + bvb.size);
  }
  if (!tau_spread_obj.is_none()) {
    Arr ts = tau_spread_obj.cast<Arr>();
    auto tsb = ts.request();
    tau_spread.assign(static_cast<const double*>(tsb.ptr), static_cast<const double*>(tsb.ptr) + tsb.size);
    bool any = false;
    for (double v : tau_spread) any |= v != 0.0;
    if (!any) tau_spread.clear();
  }
  const bool prior = tau_mode != 0 && prior_lambda > 0.0;

  Noise nz;
  {
    auto s0b = sigma0_sq_in.request(), gb = g_in.request(), cb = c_in.request();
    const double* s0 = static_cast<const double*>(s0b.ptr);
    const double* g = static_cast<const double*>(gb.ptr);
    const double* c = static_cast<const double*>(cb.ptr);
    nz.s0.resize(D); nz.g.assign(g, g + D); nz.c.assign(c, c + D);
    for (int d = 0; d < D; ++d) {
      nz.s0[d] = sigma0_mult * sigma0_mult * s0[d];
      nz.quad |= c[d] != 0.0;
    }
  }

  // saturated-pattern libraries for the R1 re-fit
  std::map<std::string, int> pattern_id;
  std::vector<Lib> pattern_lib;
  std::vector<int> event_pattern(N, -1);
  if (E.present && E.sat_level > 0.0) {
    for (int i = 0; i < N; ++i) {
      const double* y = raw + static_cast<size_t>(i) * D;
      bool any = false;
      for (int d = 0; d < D; ++d) any |= y[d] >= E.sat_level;
      if (!any) continue;
      std::string key(D, '0');
      for (int d = 0; d < D; ++d) if (y[d] >= E.sat_level) key[d] = '1';
      auto it = pattern_id.find(key);
      if (it == pattern_id.end()) {
        std::vector<int> keep;
        for (int d = 0; d < D; ++d) if (key[d] == '0') keep.push_back(d);
        it = pattern_id.emplace(key, static_cast<int>(pattern_lib.size())).first;
        pattern_lib.push_back(build_lib(S1, A1, F, D, J, keep, true));
      }
      event_pattern[i] = it->second;
    }
  }

  py::array_t<double> x_np({N, F}), x0_np({N, F}), alpha_np(N), alpha0_np(N), chi2_np(N);
  py::array_t<std::int64_t> idx_np(N);
  double* xo = static_cast<double*>(x_np.request().ptr);
  double* x0o = static_cast<double*>(x0_np.request().ptr);
  double* ao = static_cast<double*>(alpha_np.request().ptr);
  double* a0o = static_cast<double*>(alpha0_np.request().ptr);
  double* co = static_cast<double*>(chi2_np.request().ptr);
  std::int64_t* io = static_cast<std::int64_t*>(idx_np.request().ptr);
  const int n_chunks = (N + kChunk - 1) / kChunk;

  {
    py::gil_scoped_release release;
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#pragma omp parallel
    {
      Chunk c;
      c.size(kChunk, F, D, J, T);
      std::vector<double> ysub(D);
#pragma omp for schedule(dynamic, 1)
      for (int ch = 0; ch < n_chunks; ++ch) {
        const int i0 = ch * kChunk;
        const int B = std::min(kChunk, N - i0);
        const double* Y = raw + static_cast<size_t>(i0) * D;
        // ---- R1 start
        const double* Y1 = Y;
        if (whiten) {
          for (int bi = 0; bi < B; ++bi)
            for (int d = 0; d < D; ++d) c.YW[bi * D + d] = Y[static_cast<size_t>(bi) * D + d] * sw0[d];
          Y1 = c.YW.data();
          gemm_rm(B, F, D, Y, D, Pt0.data(), F, c.U0.data(), F);   // unweighted OLS for the prior SD
        }
        gemm_rm(B, F, D, Y1, D, Pt.data(), F, c.U.data(), F);
        if (has_af) {
          gemm_rm(B, J, D, Y1, D, Rt.data(), J, c.NUM.data(), J);
          for (int q = 0; q < B * F; ++q) c.PU[q] = std::max(c.U[q], 0.0);
          gemm_rm(B, D, F, c.PU.data(), F, S1, D, c.FIT.data(), D);
          for (int q = 0; q < B * D; ++q) c.FIT[q] = Y1[q] - c.FIT[q];   // residual
          gemm_rm(B, J, D, c.FIT.data(), D, Rt.data(), J, c.RR.data(), J);
        }
        for (int bi = 0; bi < B; ++bi) {
          const int i = i0 + bi;
          double* x0 = x0o + static_cast<size_t>(i) * F;
          double alpha = 0.0;
          int idx = -1;
          if (has_af) {
            const double* res = &c.FIT[bi * D];
            const double base_r = dot(res, res, D) + 1e-6;
            r1_score(r1lib, E, &c.U[bi * F], base_r, &c.NUM[bi * J], &c.RR[bi * J], c.sc, x0, alpha, idx);
            const int pat = event_pattern[i];
            if (pat >= 0 && pattern_lib[pat].ok) {
              const double* y = Y + static_cast<size_t>(bi) * D;
              int m = 0;
              for (int d = 0; d < D; ++d) if (y[d] < E.sat_level) ysub[m++] = whiten ? y[d] * sw0[d] : y[d];
              r1_event(pattern_lib[pat], E, ysub.data(), c.sc, x0, alpha, idx);
            }
          } else {
            std::memcpy(x0, &c.U[bi * F], sizeof(double) * F);
          }
          a0o[i] = alpha;
          io[i] = idx;
          c.ALPHA[bi] = alpha;
          c.IDX[bi] = idx;
          std::memcpy(&c.X[bi * F], x0, sizeof(double) * F);
          std::memcpy(&c.THETA[bi * F], x0, sizeof(double) * F);
          // prior variance of the negative per channel
          if (prior && !start_only) {
            double* t2 = &c.T2[bi * F];
            const double* u = whiten ? &c.U0[bi * F] : &c.U[bi * F];
            if (tau_mode == 1) {
              const double num0 = whiten ? dot(Y + static_cast<size_t>(bi) * D, &full.r[0], D) : c.NUM[bi * J];
              const double k0 = std::max(num0 / full.k_denom[0], 0.0);
              const double* cv = &E.null_var[0];
              for (int f = 0; f < F; ++f) {
                c.POS[f] = u[f] - k0 * full.vT[f];
                t2[f] = cv[f] + k0 * (cv[F + f] + k0 * cv[2 * F + f]);
              }
            } else {
              for (int f = 0; f < F; ++f) {
                c.POS[f] = u[f];
                t2[f] = bead_var[f];
              }
            }
            if (!tau_spread.empty())
              for (int a = 0; a < F; ++a) {
                const double pa = std::max(c.POS[a], 0.0);
                if (pa == 0.0) continue;
                for (int f = 0; f < F; ++f) t2[f] += pa * tau_spread[a * F + f];
              }
            for (int f = 0; f < F; ++f) {
              const double tau = std::sqrt(t2[f]);
              t2[f] = std::max(tau * tau, kTiny);
            }
          }
        }
        if (start_only) {
          for (int bi = 0; bi < B; ++bi) {
            const int i = i0 + bi;
            std::memcpy(xo + static_cast<size_t>(i) * F, &c.X[bi * F], sizeof(double) * F);
            ao[i] = c.ALPHA[bi];
            co[i] = 0.0;
          }
          continue;
        }
        // ---- penalised GLS refit
        const int iters = std::max(1, n_iter);
        for (int it = 0; it < iters; ++it) {
          for (int q = 0; q < B * F; ++q) c.PU[q] = std::max(c.X[q], 0.0);
          gemm_rm(B, D, F, c.PU.data(), F, S, D, c.MU.data(), D);
          for (int bi = 0; bi < B; ++bi) {
            const double* y = Y + static_cast<size_t>(bi) * D;
            double* mu = &c.MU[bi * D];
            double* w = &c.W[bi * D];
            if (has_af) {
              const double al = std::max(c.ALPHA[bi], 0.0);
              const double* arow = A + static_cast<size_t>(c.IDX[bi]) * D;
              for (int d = 0; d < D; ++d) mu[d] += al * arow[d];
            }
            for (int d = 0; d < D; ++d) {
              const double mp = std::max(mu[d], 0.0);
              double v = mp * nz.g[d] + nz.s0[d];
              if (nz.quad) v += nz.c[d] * mp * mp;
              w[d] = 1.0 / std::max(v, kTiny);
            }
            if (weight_ceiling > 0.0) {
              bool any = false;
              for (int d = 0; d < D; ++d) any |= y[d] >= 0.999 * weight_ceiling;
              if (any) {
                std::memcpy(c.wsort.data(), w, sizeof(double) * D);
                const double small = 1e-6 * median_of(c.wsort, D);
                for (int d = 0; d < D; ++d) if (y[d] >= 0.999 * weight_ceiling) w[d] = small;
              }
            }
            for (int d = 0; d < D; ++d) c.WY[bi * D + d] = w[d] * y[d];
          }
          gemm_rm(B, T, D, c.W.data(), D, O.data(), T, c.NSS.data(), T);
          gemm_rm(B, F, D, c.WY.data(), D, St.data(), F, c.RHS.data(), F);
          if (batch_lanes == 0) {
            for (int bi = 0; bi < B; ++bi) {
              const double* y = Y + static_cast<size_t>(bi) * D;
              const double* w = &c.W[bi * D];
              double* x = &c.X[bi * F];
              const double* pen = nullptr;
              if (prior) {
                const double* t2 = &c.T2[bi * F];
                for (int f = 0; f < F; ++f) {
                  const double z = std::max(x[f], 0.0) / std::sqrt(std::max(t2[f], kVarFloor)) * kInvSqrt2;
                  c.pen[f] = prior_lambda * std::erfc(z) / t2[f];
                }
                pen = c.pen.data();
              }
              const double* nss = &c.NSS[static_cast<size_t>(bi) * T];
              const double* rhs = &c.RHS[bi * F];
              bool ok;
              double a_new = 0.0;
              if (has_af) {
                const double* arow = A + static_cast<size_t>(c.IDX[bi]) * D;
                std::fill(c.nsa.begin(), c.nsa.end(), 0.0);
                double naa = 0.0, rhs_a = 0.0;
                for (int d = 0; d < D; ++d) {
                  const double wa = w[d] * arow[d];
                  naa += wa * arow[d];
                  rhs_a += wa * y[d];
                  const double* sd = &St[d * F];
  #pragma omp simd
                  for (int f = 0; f < F; ++f) c.nsa[f] += wa * sd[f];
                }
                ok = solve_event(c, F, nss, rhs, true, c.nsa.data(), naa, rhs_a, pen);
                if (ok) a_new = c.b[F];
                if (ok && a_new < 0.0) {
                  ok = solve_event(c, F, nss, rhs, false, nullptr, 0.0, 0.0, pen);
                  a_new = 0.0;
                }
              } else {
                ok = solve_event(c, F, nss, rhs, false, nullptr, 0.0, 0.0, pen);
              }
              if (!ok) continue;   // keeps the previous estimate
              c.ALPHA[bi] = a_new;
              double* th = &c.THETA[bi * F];
              for (int f = 0; f < F; ++f) {
                th[f] = c.b[f];
                double v = c.b[f];
                if (pen) v /= std::max(1.0 - pen[f] * c.inv_diag[f], 1e-3);
                x[f] = v;
              }
            }
          } else {
            // several events at once: per-event ridge and AF column, then the batched solves
            for (int bi = 0; bi < B; ++bi) {
              const double* x = &c.X[bi * F];
              if (prior) {
                const double* t2 = &c.T2[bi * F];
                double* pe = &c.PENB[bi * F];
                for (int f = 0; f < F; ++f) {
                  const double z = std::max(x[f], 0.0) / std::sqrt(std::max(t2[f], kVarFloor)) * kInvSqrt2;
                  pe[f] = prior_lambda * std::erfc(z) / t2[f];
                }
              }
              c.EV[bi] = bi;
            }
            if (has_af) {
              for (int bi = 0; bi < B; ++bi) {
                const double* y = Y + static_cast<size_t>(bi) * D;
                const double* w = &c.W[bi * D];
                const double* arow = A + static_cast<size_t>(c.IDX[bi]) * D;
                double* wa = &c.WA[bi * D];
                double naa = 0.0, rhs_a = 0.0;
                for (int d = 0; d < D; ++d) {
                  wa[d] = w[d] * arow[d];
                  naa += wa[d] * arow[d];
                  rhs_a += wa[d] * y[d];
                }
                c.NAAB[bi] = naa;
                c.RHSAB[bi] = rhs_a;
              }
              gemm_rm(B, F, D, c.WA.data(), D, St.data(), F, c.NSAB.data(), F);
            }
            const double* PE = prior ? c.PENB.data() : nullptr;
            auto run = [&](bool with_af, int n) {
              if (batch_lanes == 8)
                batch_solve<8>(F, with_af, T, c.EV.data(), n, c.NSS.data(), c.RHS.data(), c.NSAB.data(),
                               c.NAAB.data(), c.RHSAB.data(), PE, c.BOUT.data(), c.INVD.data(), c.OKV.data(), c.sbuf);
              else
                batch_solve<4>(F, with_af, T, c.EV.data(), n, c.NSS.data(), c.RHS.data(), c.NSAB.data(),
                               c.NAAB.data(), c.RHSAB.data(), PE, c.BOUT.data(), c.INVD.data(), c.OKV.data(), c.sbuf);
            };
            run(has_af, B);
            if (has_af) {   // a negative AF abundance: re-solve those events without the AF column
              int n2 = 0;
              for (int bi = 0; bi < B; ++bi)
                if (c.OKV[bi] && c.BOUT[static_cast<size_t>(bi) * (F + 1) + F] < 0.0) c.EV[n2++] = bi;
              if (n2 > 0) run(false, n2);
            }
            for (int bi = 0; bi < B; ++bi) {
              if (!c.OKV[bi]) continue;   // keeps the previous estimate
              const double* bo = &c.BOUT[static_cast<size_t>(bi) * (F + 1)];
              const double* inv = &c.INVD[static_cast<size_t>(bi) * F];
              const double* pen = prior ? &c.PENB[bi * F] : nullptr;
              c.ALPHA[bi] = has_af ? bo[F] : 0.0;
              double* th = &c.THETA[bi * F];
              double* x = &c.X[bi * F];
              for (int f = 0; f < F; ++f) {
                th[f] = bo[f];
                double v = bo[f];
                if (pen) v /= std::max(1.0 - pen[f] * inv[f], 1e-3);
                x[f] = v;
              }
            }
          }
        }
        // chi2 of the last solve: weighted residual of the unpenalised-scale coefficients
        gemm_rm(B, D, F, c.THETA.data(), F, S, D, c.MU.data(), D);
        for (int bi = 0; bi < B; ++bi) {
          const int i = i0 + bi;
          const double* y = Y + static_cast<size_t>(bi) * D;
          const double* w = &c.W[bi * D];
          const double* fit = &c.MU[bi * D];
          const double al = has_af ? c.ALPHA[bi] : 0.0;
          const double* arow = has_af ? A + static_cast<size_t>(c.IDX[bi]) * D : nullptr;
          double chi2 = 0.0;
          for (int d = 0; d < D; ++d) {
            const double r = y[d] - fit[d] - (arow ? al * arow[d] : 0.0);
            chi2 += w[d] * r * r;
          }
          std::memcpy(xo + static_cast<size_t>(i) * F, &c.X[bi * F], sizeof(double) * F);
          ao[i] = c.ALPHA[bi];
          co[i] = chi2;
        }
      }
    }
  }
  py::dict out;
  out["unmixed"] = x_np;
  out["af_idx"] = idx_np;
  out["af_scale"] = alpha_np;
  out["chi2"] = chi2_np;
  out["start_unmixed"] = x0_np;
  out["start_af_scale"] = alpha0_np;
  out["prior"] = prior;
  return out;
}

static int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

PYBIND11_MODULE(_gls_m2p_kernel, m) {
  m.doc() = "M2p (penalised GLS with the positivity prior) from raw events, R1 start included";
  m.def("unmix_m2p", &unmix_m2p, py::arg("raw"), py::arg("spectra"), py::arg("af_spectra"),
        py::arg("envelope"), py::arg("sigma0_sq"), py::arg("g"), py::arg("c"), py::arg("sigma0_mult") = 1.0,
        py::arg("bead_var") = py::none(), py::arg("tau_spread") = py::none(), py::arg("prior_lambda") = 1.0,
        py::arg("n_iter") = 2, py::arg("weight_ceiling") = 0.0, py::arg("n_threads") = 0,
        py::arg("start_only") = false, py::arg("af_choice_weights") = py::none(),
        py::arg("batch_lanes") = 0);
  m.def("max_threads", &max_threads);
}
