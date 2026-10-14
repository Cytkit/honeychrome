// af_negative_term.cpp
// ---------------------------------------------------------------------------
// Positivity-weighted fluorophore term (see af_negative_term.hpp).
// ---------------------------------------------------------------------------
#include "af_negative_term.hpp"

#include <cmath>
#include <cstddef>

void positivity_weight_event(const double* position, const double* var, double* weight, int n_fluor) {
  static const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  for (int f = 0; f < n_fluor; ++f) {
    const double x = position[f] > 0.0 ? position[f] : 0.0;
    const double v = var[f] > 1e-24 ? var[f] : 1e-24;
    weight[f] = std::erfc(x / std::sqrt(v) * inv_sqrt2);
  }
}

void positivity_fluor_term_event(const double* u, const double* k, const double* v_t,
                                 const double* weight, double* out, int n_af, int n_fluor) {
  // Base: every channel at its AF-free value, so the term is proportional.
  double base = 1e-6;
  for (int f = 0; f < n_fluor; ++f) base += weight[f] * u[f] * u[f];
  const double inv_base = 1.0 / base;

  for (int j = 0; j < n_af; ++j) {
    const double kj = k[j];
    const double* vj = v_t + static_cast<std::size_t>(j) * n_fluor;
    double cross = 0.0, curv = 0.0;
    for (int f = 0; f < n_fluor; ++f) {
      const double wv = weight[f] * vj[f];
      cross += wv * u[f];
      curv += wv * vj[f];
    }
    const double e = base - 2.0 * kj * cross + kj * kj * curv;
    out[j] = (e > 0.0 ? e : 0.0) * inv_base;
  }
}
