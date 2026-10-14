// af_negative_term.hpp
// ---------------------------------------------------------------------------
// Fluorophore term of the positivity-weighted (negative-envelope) AF score,
// for one event and every AF candidate. See
// autospectral_functions._positivity_fluor_term for the definition; the
// joint unmixing kernel's AF scoring uses these per event.
// ---------------------------------------------------------------------------
#pragma once

// 1 - p per channel, where p = erf(max(x, 0) / sqrt(2 var)) is the share of
// the negative's distribution lying closer to zero than x
// (autospectral_functions.positivity_weight). var is floored at 1e-24.
//   position, var, weight : n_fluor entries.
void positivity_weight_event(const double* position, const double* var, double* weight, int n_fluor);

// Proportional weighted fluorophore error of every candidate,
//   out[j] = sum_f weight_f (u_f - k_j v_jf)^2 / (sum_f weight_f u_f^2 + 1e-6),
// expanded as a quadratic in k_j as the NumPy term is.
//   u      (n_fluor)            AF-free unmix
//   k      (n_af)               abundance of each candidate
//   v_t    (n_af, n_fluor)      in-span projection of each candidate, row-major
//   weight (n_fluor)            positivity weight times the fluorophore error weight
//   out    (n_af)
void positivity_fluor_term_event(const double* u, const double* k, const double* v_t,
                                 const double* weight, double* out, int n_af, int n_fluor);
