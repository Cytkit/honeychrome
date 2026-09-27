// core_joint_unmix.hpp
// ---------------------------------------------------------------------------
// Armadillo-only core for the AutoSpectral joint AF + fluorophore-variant
// per-cell unmixing pipeline, ported from AutoSpectralRcpp's
// unmix_autospectral_joint_cpp() (unmix_autospectral_joint_pipeline.cpp).
// All Rcpp types (CharacterVector, Rcpp::List, Rcpp::Nullable) are replaced
// with plain Armadillo/STL types so this file depends only on Armadillo and
// the standard library. Wrapped for Python by autospectral_opt_pybind.cpp.
// ---------------------------------------------------------------------------
#pragma once

#include <armadillo>
#include <string>
#include <vector>

// One optimizable fluorophore's variant data, as discovered by
// autospectral_optimization_functions.py::discover_fluor_variants() (Python
// port of get.fluor.variants()) and assembled by
// autospectral_optimization_functions.py::unmix_autospectral_optimization().
struct FluorVariantInput {
  std::string name;        // fluorophore label, must match a row of `spectra`
  arma::mat    v_mats;      // n_variants x D — candidate mixing (spectrum) matrix
  arma::mat    delta_obs;   // n_variants x D — (v_mats - reference row); same
                             // quantity as R's delta.list[[fl]], used for the
                             // covariance-propagated leakage weight.
};

// Joint per-cell AF + fluorophore-variant unmixing.
//
// Parameters
// ----------
// raw_data_in      : N x D  — raw fluorescence events.
// spectra          : F x D  — reference fluorophore spectra (no AF row).
// af_spectra       : nAF x D — AF candidate spectra, nAF >= 2.
// fluor_names      : length F, row order matching `spectra`.
// pos_thresholds   : length F — per-fluorophore unmixed-space positivity
//                     threshold, gates whether joint variant optimisation is
//                     attempted for a cell.
// variants         : one entry per *optimizable* fluorophore. Empty vector ->
//                     AF-only mode (no joint variant optimisation).
// n_passes         : joint variant-optimisation passes per cell.
// n_threads        : OpenMP thread count.
// cell_weight      : per-cell detector weighting on/off.
// noise_floor      : nullptr or empty -> fill 125.0 everywhere; length 1 ->
//                     broadcast that scalar to all D detectors; length D ->
//                     used as-is.
// alpha            : residual-vs-leakage balance in the joint score.
// collinear_thresh : cosine threshold for structurally-collinear fluorophore
//                     pairs.
// joint_pair_resolution : retry conflicting candidates against collinear
//                     partners after the first commit pass.
// n_af_passes      : AF refinement passes per cell.
// refine_af_quantile : fraction of cells carried into extra AF passes.
// exact_variant_scan : score candidate variants with the closed-form
//                     single-endmember swap rather than the fixed-abundance
//                     residual approximation. Candidates are still verified
//                     by a full re-solve before acceptance. Ignored when
//                     cell_weight is true.
//
// Returns
// -------
// N x (F+2) matrix: [fluor_1 .. fluor_F | AF abundance | AF index (1-based)].
arma::mat unmix_autospectral_joint_core(
    const arma::mat&                       raw_data_in,
    const arma::mat&                       spectra,
    const arma::mat&                       af_spectra,
    const std::vector<std::string>&        fluor_names,
    const arma::vec&                       pos_thresholds,
    const std::vector<FluorVariantInput>&  variants,
    int                                     n_passes               = 1,
    int                                     n_threads               = 1,
    bool                                    cell_weight             = false,
    const arma::vec*                       noise_floor             = nullptr,
    double                                  alpha                   = 0.5,
    double                                  collinear_thresh        = 0.5,
    bool                                    joint_pair_resolution   = true,
    int                                     n_af_passes             = 1,
    double                                  refine_af_quantile      = 0.5,
    bool                                    exact_variant_scan      = false
);
