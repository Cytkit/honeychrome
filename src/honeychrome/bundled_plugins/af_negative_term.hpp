// af_negative_term.hpp
// ---------------------------------------------------------------------------
// Fluorophore term of the guarded negative-envelope AF score, for every
// event and AF candidate. See autospectral_functions._guarded_fluor_term for
// the definition; this is the same computation without the (events x
// candidates x fluorophores) temporary.
// ---------------------------------------------------------------------------
#pragma once

// All arrays are C-contiguous float64.
//   u      (n_events, n_fluor)   AF-free unmix
//   k      (n_events, n_af)      abundance of each candidate
//   v_t    (n_af, n_fluor)       in-span projection of each candidate
//   w      (n_fluor)             fluorophore error weights
//   upper  (n_events, n_fluor)   per-event upper boundary
//   floor_ (n_events, n_fluor)   per-event lower penalty boundary
//   keep   (n_events, n_fluor)   how negative each channel is, 1 negative to 0 positive
//   out    (n_events, n_af)      proportional fluorophore error
// n_threads <= 0 uses every available thread.
//
// guarded_fluor_term_event scores one event (u, upper, floor_, keep are one
// row; k and out have n_af entries).
void guarded_fluor_term_event(const double* u, const double* k, const double* v_t,
                              const double* w, const double* upper, const double* floor_,
                              const double* keep, double hyper_weight, double* out,
                              int n_af, int n_fluor);

void guarded_fluor_term_core(const double* u, const double* k, const double* v_t,
                             const double* w, const double* upper, const double* floor_,
                             const double* keep, double hyper_weight, double* out,
                             int n_events, int n_af, int n_fluor, int n_threads);

// How negative each channel of one event is (autospectral_functions._guard_weight):
// with d = (position - upper) / max(sd, 1e-12), 1 for d <= -half_width, 0 for
// d >= half_width and a cubic smoothstep in between; half_width <= 0 is a step
// at the boundary.
void guard_weight_event(const double* position, const double* upper, const double* sd,
                        double half_width, double* keep, int n_fluor);
