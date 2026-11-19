// kernel_linalg.hpp
// ---------------------------------------------------------------------------
// Small dense linear algebra shared by the sources of the _gls_m2p_kernel
// extension (gls_m2p_kernel.cpp, af_bend_kernel.cpp): a dot product, a
// row-major wrapper of BLAS dgemm and an in-place Cholesky with its solve.
// Build with -DGLS_DGEMM=<symbol> -DGLS_BLAS_INT=<int type> for a BLAS whose
// dgemm is not plain dgemm_ with 32-bit integers.
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>

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

namespace kernel_linalg {

inline double dot(const double* a, const double* b, int n) {
  double s = 0.0;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// Row-major C (m x n) = A (m x k) B (k x n) + beta C.
inline void gemm_rm(int m, int n, int k, const double* A, int lda, const double* B, int ldb, double* C, int ldc,
                    double beta = 0.0) {
  if (m == 0 || n == 0) return;
  const char t = 'N';
  const GLS_BLAS_INT M = n, N = m, K = k, LA = ldb, LB = lda, LC = ldc;
  const double one = 1.0;
  GLS_DGEMM(&t, &t, &M, &N, &K, &one, B, &LA, A, &LB, &beta, C, &LC);
}

// Cholesky of a p x p SPD matrix in place (lower triangle, row-major). False if not SPD.
inline bool cholesky(double* a, int p) {
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
inline void chol_solve(const double* l, int p, double* b) {
  for (int i = 0; i < p; ++i) b[i] = (b[i] - dot(&l[i * p], b, i)) / l[i * p + i];
  for (int i = p - 1; i >= 0; --i) {
    double s = b[i];
    for (int k = i + 1; k < p; ++k) s -= l[k * p + i] * b[k];
    b[i] = s / l[i * p + i];
  }
}

}  // namespace kernel_linalg
