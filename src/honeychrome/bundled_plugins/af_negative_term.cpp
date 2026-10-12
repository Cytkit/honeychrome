// af_negative_term.cpp
// ---------------------------------------------------------------------------
// Guarded negative-envelope fluorophore term (see af_negative_term.hpp). The
// per-event function is shared by the batch entry point, where events are
// split across threads with OpenMP, and by the joint unmixing kernel's AF
// scoring.
// ---------------------------------------------------------------------------
#include "af_negative_term.hpp"

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

void guarded_fluor_term_event(const double* u, const double* k, const double* v_t,
                              const double* w, const double* upper, const double* floor_,
                              const double* keep, double hyper_weight, double* out,
                              int n_af, int n_fluor) {
  // Base: every channel at its AF-free value, so the term is proportional.
  double base = 1e-6;
  for (int f = 0; f < n_fluor; ++f) {
    const double g = keep[f];
    const double m = u[f] < upper[f] ? u[f] : upper[f];
    double term = g * m * m;
    if (g < 1.0) term += (1.0 - g) * upper[f] * upper[f];
    base += w[f] * term;
  }
  const double inv_base = 1.0 / base;

  for (int j = 0; j < n_af; ++j) {
    const double kj = k[j];
    const double* vj = v_t + static_cast<std::size_t>(j) * n_fluor;
    double sum = 0.0;
    for (int f = 0; f < n_fluor; ++f) {
      const double x = u[f] - kj * vj[f];
      const double g = keep[f];
      // Negative part: towards 0, capped at the boundary.
      const double capped = x < upper[f] ? x : upper[f];
      double term = g * capped * capped;
      // Positive part: penalty for being pulled below the boundary.
      if (g < 1.0) {
        double pulled = upper[f] - x;
        pulled = pulled > 0.0 ? pulled : 0.0;
        term += (1.0 - g) * pulled * pulled;
      }
      double below = floor_[f] - x;
      below = below > 0.0 ? below : 0.0;
      sum += w[f] * (term + hyper_weight * below * below);
    }
    out[j] = sum * inv_base;
  }
}

void guarded_fluor_term_core(const double* u, const double* k, const double* v_t,
                             const double* w, const double* upper, const double* floor_,
                             const double* keep, double hyper_weight, double* out,
                             int n_events, int n_af, int n_fluor, int n_threads) {
#ifdef _OPENMP
  const int threads = n_threads > 0 ? n_threads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(threads)
#else
  (void)n_threads;
#endif
  for (int b = 0; b < n_events; ++b) {
    const std::size_t row_f = static_cast<std::size_t>(b) * n_fluor;
    const std::size_t row_a = static_cast<std::size_t>(b) * n_af;
    guarded_fluor_term_event(u + row_f, k + row_a, v_t, w, upper + row_f, floor_ + row_f,
                             keep + row_f, hyper_weight, out + row_a, n_af, n_fluor);
  }
}

void guard_weight_event(const double* position, const double* upper, const double* sd,
                        double half_width, double* keep, int n_fluor) {
  for (int f = 0; f < n_fluor; ++f) {
    if (half_width <= 0.0) {
      keep[f] = position[f] <= upper[f] ? 1.0 : 0.0;
      continue;
    }
    const double d = (position[f] - upper[f]) / std::max(sd[f], 1e-12);
    const double t = std::min(std::max((d / half_width + 1.0) * 0.5, 0.0), 1.0);
    keep[f] = 1.0 - t * t * (3.0 - 2.0 * t);
  }
}
