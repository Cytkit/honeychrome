// core_joint_unmix.cpp
// ---------------------------------------------------------------------------
// See core_joint_unmix.hpp for the interface contract.
//
// Port of AutoSpectralRcpp's unmix_autospectral_joint_cpp()
// (unmix_autospectral_joint_pipeline.cpp). The numerical algorithm is
// unchanged; only the Rcpp input marshalling differs:
//   - Rcpp::List variants_list/delta_list -> std::vector<FluorVariantInput>
//   - Rcpp::CharacterVector fluor_names   -> std::vector<std::string>
//   - Rcpp::Nullable<NumericVector> noise_floor -> const arma::vec* (nullable)
//   - Rcpp::stop(...)                    -> std::invalid_argument
// ---------------------------------------------------------------------------
#include "core_joint_unmix.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace arma;

namespace {

double quantile_type7(arma::vec x, double p) {
  const uword n = x.n_elem;
  if (n <= 1) return n == 1 ? x[0] : 0.0;
  x = arma::sort(x);
  const double h  = (n - 1) * p;
  const uword  lo = (uword)std::floor(h);
  const uword  hi = std::min(lo + 1, n - 1);
  return x[lo] + (h - lo) * (x[hi] - x[lo]);
}

// ---------------------------------------------------------------------------
// Precomputed data for a single optimisable endmember
// ---------------------------------------------------------------------------
struct FluorPrecomp {
  bool    active     = false;
  int     master_idx = -1;

  mat  v_mats;      // n_variants x D  – candidate mixing matrix
  mat  delta;       // n_variants x D  – (v_mats - master_row), pre-centred

  // Leakage prediction via the other-endmember pseudoinverse
  mat  v_lib;       // (F-1) x n_variants – how each variant leaks into others
  vec  w_leakage;   // F-1               – per-other-endmember cov-propagated weights

  // Rank-1 residual update helpers
  mat  r_lib;       // D x n_variants  – residual component of each variant delta
  mat  r_lib_sq;    // D x n_variants  – r_lib .* r_lib (for weighted self-dots)
  vec  r_dots;      // n_variants      – dot(r_lib(:,v), r_lib(:,v))

  // Exact single-swap (Frisch-Waugh) helpers. M is the projector orthogonal to
  // the other F-1 endmembers, so everything the rest of the panel can explain
  // is fixed and independent of which variant is swapped in. What remains is a
  // one-dimensional fit, solvable in closed form.
  double c_ff = 0.0;  //                 – [(S S^T)^-1]_ff, so 1/c_ff = ||M s_f||^2
  vec  u_f;           // F-1             – U_nof applied to the master spectrum
  mat  g_lib;         // D x n_variants  – M applied to each variant delta
  vec  q;             // n_variants      – ||M (s_f + delta_v)||^2

  uword n_variants = 0;
};

}  // namespace

arma::mat unmix_autospectral_joint_core(
    const arma::mat&                       raw_data_in,
    const arma::mat&                       spectra,
    const arma::mat&                       af_spectra,
    const std::vector<std::string>&        fluor_names,
    const arma::vec&                       pos_thresholds,
    const std::vector<FluorVariantInput>&  variants,
    int    n_passes,
    int    n_threads,
    bool   cell_weight,
    const arma::vec* noise_floor,
    double alpha,
    double collinear_thresh,
    bool   joint_pair_resolution,
    int    n_af_passes,
    double refine_af_quantile,
    bool   exact_variant_scan
) {
  const mat raw_data = raw_data_in.t();   // D x N
  const uword N   = raw_data.n_cols;
  const uword F   = spectra.n_rows;
  const uword D   = spectra.n_cols;
  const uword nAF = af_spectra.n_rows;

  if (n_af_passes < 1)
    throw std::invalid_argument("n_af_passes must be >= 1.");
  if (refine_af_quantile < 0.0 || refine_af_quantile > 1.0)
    throw std::invalid_argument("refine_af_quantile must be between 0 and 1.");

  // Handle vectorized noise floor parsing and fallbacks
  arma::vec current_noise_floor;
  if (noise_floor != nullptr && !noise_floor->is_empty())
    current_noise_floor = *noise_floor;
  if (current_noise_floor.is_empty()) {
    current_noise_floor.set_size(D);
    current_noise_floor.fill(125.0); // Default fallback value
  } else if (current_noise_floor.n_elem == 1) {
    double scalar_floor = current_noise_floor[0];
    current_noise_floor.set_size(D);
    current_noise_floor.fill(scalar_floor);
  }

  // =========================================================================
  // SECTION 1 – Global pre-computations
  // =========================================================================

  // Global weight vector: 1 / max(mean detector signal, noise_floor).
  // raw_data is D x N, so mean over columns gives a D-vector.
  vec w_global(D);
  vec sqrt_w_global(D);
  if (cell_weight) {
    const vec col_means = mean(raw_data, 1);
    for (uword d = 0; d < D; ++d) {
      w_global[d]      = 1.0 / std::max(col_means[d], current_noise_floor[d]);
      sqrt_w_global[d] = std::sqrt(w_global[d]);
    }
  } else {
    w_global.ones();
    sqrt_w_global.ones();
  }

  // Global weighted pseudoinverse P_w = (S W S^T)^{-1} S W, shape F x D.
  const mat spectra_w  = spectra.each_row() % sqrt_w_global.t();  // F x D
  const mat SST_global = spectra_w * spectra_w.t();               // F x F, constant across cells whenever cell_weight == false
  mat P = solve(SST_global, spectra_w);                            // F x D
  P.each_row() %= sqrt_w_global.t();

  // Unweighted inverse Gram of the panel. Its f-th diagonal entry is
  // 1 / ||M_f s_f||^2, where M_f projects orthogonal to the other F-1
  // endmembers, and that is the quantity the exact variant-swap solution is
  // written in terms of.
  mat SS_inv;
  {
    const mat SST_raw = symmatu(spectra * spectra.t());
    if (!inv_sympd(SS_inv, SST_raw)) SS_inv = pinv(SST_raw);
  }

  // Detector-major copy of af_spectra (D x nAF). Used in the per-cell loop
  // below so subtracting a selected AF spectrum from a cell's residual reads
  // a genuine contiguous column instead of transposing a (memory-strided)
  // row of af_spectra on every cell.
  const mat af_spectra_t = af_spectra.t();

  // AF helpers — computed in weighted detector space
  const mat v_lib_af  = P * af_spectra.t();                        // F   x nAF
  const mat r_lib_af  = af_spectra.t() - spectra.t() * v_lib_af;  // D   x nAF

  vec r_dots_af(nAF);
  for (uword j = 0; j < nAF; ++j) {
    const vec r_w = r_lib_af.col(j) % sqrt_w_global;
    r_dots_af[j] = dot(r_w, r_w);
  }
  // True weighted self-dot, taken before the identifiability floor below. The
  // floor belongs in the abundance denominator only: the rank-1 residual-norm
  // update has to use the unmodified value, or it stops evaluating the actual
  // residual at the abundance that was chosen.
  const vec r_dots_af_w = r_dots_af;
  // Identifiability guard: an AF variant lying almost inside the fluorophore
  // span has a vanishing out-of-span residual direction, so its abundance k
  // is not identifiable from the residual and the raw ratio explodes. Floor
  // each self-dot at a fraction of the largest candidate's self-dot, matching
  // unmix_autospectral_joint.R. For a library of hundreds to thousands of AF
  // candidates, a median-based floor is systematically far weaker than a
  // max-based one from ordinary order-statistics spread alone -- no
  // corrupted or outlier candidate required -- so it under-floors
  // near-in-span candidates broadly, inflating their abundance and producing
  // hypernegative fluorophore values in exactly the channels this guard
  // exists to protect.
  const double r_dots_scale = arma::max(r_dots_af);
  const double r_dots_floor = 0.01 * std::max(r_dots_scale, 1e-10);
  r_dots_af = clamp(r_dots_af, r_dots_floor, arma::datum::inf);

  // r_lib_af pre-scaled by w_global (one power, matching r_dots_af above),
  // so the per-cell k_j numerator for every AF candidate collapses into a
  // single gemv instead of nAF dot()s. k_j is the weighted least-squares
  // solution k = <r_j, y>_w / <r_j, r_j>_w, so numerator and denominator
  // must carry the same power of w.
  const mat r_lib_af_w = r_lib_af.each_col() % w_global;

  // Covariance-propagated AF endmember weights.
  const mat af_cov_mat = mat(P * cov(af_spectra) * P.t());
  const vec w_af = sqrt(abs(af_cov_mat.diag())) + 1e-8;

  // Weighted v_lib_af and its squared-column sums, precomputed once so the
  // per-cell fluorophore-leakage term below is a single rank-1 (quadratic)
  // update instead of an O(F x nAF) abs() pass per cell.
  const mat v_lib_af_w = v_lib_af.each_col() % w_af;              // F x nAF
  const vec c_fluor_af = (sum(v_lib_af_w % v_lib_af_w, 0)).t();   // nAF

  // Determine whether endmember variant optimisation is requested.
  const bool af_only = variants.empty();

  // =========================================================================
  // SECTION 2 – Per-endmember pre-computations
  // =========================================================================

  std::map<std::string, int> name_to_idx;
  for (size_t i = 0; i < fluor_names.size(); ++i)
    name_to_idx[fluor_names[i]] = (int)i;

  const int n_opt = af_only ? 0 : (int)variants.size();
  std::vector<FluorPrecomp> precomp(n_opt);
  std::vector<int> active_indices;

  if (!af_only) {
    for (int i = 0; i < n_opt; ++i) {
      const std::string& fname = variants[i].name;
      if (!name_to_idx.count(fname)) continue;

      FluorPrecomp& pc = precomp[i];
      pc.active     = true;
      pc.master_idx = name_to_idx[fname];
      pc.v_mats     = variants[i].v_mats;
      pc.n_variants = pc.v_mats.n_rows;
      if (pc.n_variants == 0) { pc.active = false; continue; }

      const rowvec master_row = spectra.row(pc.master_idx);

      pc.delta.set_size(pc.n_variants, D);
      for (uword v = 0; v < pc.n_variants; ++v)
        pc.delta.row(v) = pc.v_mats.row(v) - master_row;

      // Other-endmember pseudoinverse
      uvec keep(F - 1); uword ri = 0;
      for (uword r = 0; r < F; ++r) {
        if ((int)r != pc.master_idx) keep[ri++] = r;
      }

      const mat S_nof = spectra.rows(keep);             // (F-1) x D
      const mat U_nof = solve(S_nof * S_nof.t(), S_nof); // (F-1) x D

      pc.v_lib = U_nof * pc.delta.t();   // (F-1) x n_variants

      // Covariance-propagated leakage weights.
      // add a ridge (1e-4 * I) to stabilise small delta samples.
      {
        const mat& delta_obs = variants[i].delta_obs;    // n_variants x D
        mat delta_cov   = cov(delta_obs);
        delta_cov.diag() += 1e-4;                        // ridge regularisation
        const mat leakage_cov = mat(U_nof * delta_cov * U_nof.t()); // force eval
        pc.w_leakage = sqrt(abs(leakage_cov.diag())) + 1e-8;
      }

      // Rank-1 residual update helpers
      // r_lib(:,v) = delta(v)^T - S^T * (P * delta(v)^T)
      pc.r_lib  = pc.delta.t() - spectra.t() * (P * pc.delta.t());  // D x n_variants
      pc.r_lib_sq = pc.r_lib % pc.r_lib;                            // D x n_variants
      pc.r_dots.set_size(pc.n_variants);
      for (uword v = 0; v < pc.n_variants; ++v)
        pc.r_dots[v] = dot(pc.r_lib.col(v), pc.r_lib.col(v));

      // Exact single-swap helpers. M delta = delta - S_nof^T (U_nof delta), and
      // U_nof delta is already v_lib, so g_lib costs one product.
      pc.c_ff  = SS_inv(pc.master_idx, pc.master_idx);
      pc.u_f   = U_nof * master_row.t();                     // F-1
      pc.g_lib = pc.delta.t() - S_nof.t() * pc.v_lib;        // D x n_variants
      pc.q     = 1.0 / std::max(pc.c_ff, 1e-12)
        + 2.0 * (master_row * pc.g_lib).t()
        + sum(pc.g_lib % pc.g_lib, 0).t();

      // q_v is the reciprocal of the swapped panel's inverse-Gram diagonal for
      // this endmember, so a small q_v means the variant has driven it toward
      // collinearity with the rest of the panel and its abundance is no longer
      // identifiable. Floor it on the same principle as r_dots_af above.
      pc.q = clamp(pc.q, 1e-4 / std::max(pc.c_ff, 1e-12), arma::datum::inf);

      active_indices.push_back(i);
    }
  }

  // =========================================================================
  // SECTION 2B – Structural collinearity precompute
  // =========================================================================
  arma::umat is_collinear(F, F, fill::zeros);
  if (!af_only) {
    const int n_active_pre = (int)active_indices.size();
    for (int a = 0; a < n_active_pre; ++a) {
      for (int b = a + 1; b < n_active_pre; ++b) {
        const int fa = precomp[active_indices[a]].master_idx;
        const int fb = precomp[active_indices[b]].master_idx;
        const double c = std::abs(dot(P.row(fa), P.row(fb))) /
          (norm(P.row(fa)) * norm(P.row(fb)) + 1e-12);
        if (c > collinear_thresh) {
          is_collinear(fa, fb) = 1;
          is_collinear(fb, fa) = 1;
        }
      }
    }
  }

  // Largest per-endmember variant count across all active fluorophores, used
  // to pre-size the per-cell candidate-scan buffers below so cross_v / drsq_v
  // / g_cur never need to grow or shrink when moving between endmembers with
  // different variant counts within the same cell.
  uword max_n_variants = 1;
  for (const auto& pc : precomp)
    if (pc.active) max_n_variants = std::max(max_n_variants, pc.n_variants);

  // The exact swap solution is derived in the unweighted inner product, and
  // U_nof above is built unweighted, so per-cell detector weighting falls back
  // to the incremental residual scan.
  const bool use_exact = exact_variant_scan && !cell_weight;

  // =========================================================================
  // SECTION 3 – parallel loop
  // =========================================================================
  mat result(N, F + 2, fill::zeros);
  vec  af_abundance_vec(N, fill::zeros);
  uvec af_index_vec(N, fill::zeros);
  mat  resid_mat(D, N, fill::zeros);
  uvec still_active(N, fill::zeros);
  double af_refine_cutoff = 0.0;

#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif

#pragma omp parallel
{
  // Thread-local buffers. Declared `static thread_local` (matching the
  // pattern used by the R-side pipelines) so each OS
  // thread constructs them ONCE, ever, and reuses the same memory across
  // every subsequent call to this function for the life of the process --
  // not just across cells within a single call. Without `static`, every
  // call re-enters this parallel region and every thread reconstructs every
  // buffer below from scratch simultaneously: a synchronised burst of
  // concurrent allocation right at region entry, which gets worse (not
  // better) as thread count rises on Windows, and which repeated per-cell
  // allocation fixes alone cannot touch since it happens once per call, not
  // once per cell.
  //
  // Because the buffers persist across calls, and different calls can pass
  // a differently-shaped panel (different F / D / nAF / n_opt /
  // max_n_variants / n_active), every buffer is explicitly (re)sized below
  // via set_size()/assign(), every call. Armadillo's set_size() is a no-op
  // when the requested size already matches the current size, so on
  // repeated calls with the same panel this costs nothing; it only
  // reallocates when the panel shape has actually changed.
  static thread_local vec resid;
  static thread_local vec resid_raw;
  static thread_local vec fluor_unmixed;
  static thread_local mat cell_S;
  static thread_local mat cell_S_F_w;
  static thread_local vec unmixed_full;
  static thread_local vec sqrt_w;
  static thread_local vec dr;
  static thread_local std::vector<int> best_v;
  static thread_local vec init_f;
  static thread_local vec base_resid_af;
  static thread_local vec resid_w_af;
  static thread_local vec cell_resid;
  static thread_local vec y_hat;
  static thread_local vec k_af_vec;
  static thread_local vec cross_af;
  static thread_local vec resid_sq_af;
  static thread_local vec presid_af;
  static thread_local vec pfluor_af;
  static thread_local vec score_af_vec;
  static thread_local vec w_af_init;
  static thread_local vec cross_fluor_af;
  static thread_local vec fluor_sq_af;
  static thread_local vec coeff_init;
  static thread_local vec other_unmixed;
  static thread_local vec trial_unmixed;
  static thread_local vec trial_resid_raw;
  static thread_local vec trial_resid;
  static thread_local mat S_F_w;

  static thread_local mat A_base;
  static thread_local vec b_base;
  static thread_local vec y_vec;
  static thread_local mat A_trial;
  static thread_local vec b_trial;
  static thread_local rowvec s_new;
  static thread_local vec col_update;
  static thread_local rowvec prev_row;

  static thread_local vec rsw;        // resid .* sqrt_w        (recomputed per pass)
  static thread_local vec w_eff;      // sqrt_w .* sqrt_w       (per cell, weighted path only)
  static thread_local vec cross_v;    // <resid, (r_v - r_cur).*w>  for all v
  static thread_local vec drsq_v;     // ||(r_v - r_cur).*w||^2      for all v
  static thread_local vec g_cur;      // <r_v.*w, r_cur.*w>          for all v

  static thread_local vec a_base;     // F   – base-panel coefficients for this cell
  static thread_local vec c_vec;      // F-1 – other endmembers with this one dropped

  // Weighted self-dots q_a[v] = ||r_v .* w||^2, cached per active endmember
  // per cell and reused across passes (weighted path only; unweighted uses the
  // static precomputed r_dots). Lazy so below-threshold endmembers cost nothing.
  static thread_local std::vector<vec>  q_by_active;
  static thread_local std::vector<char> q_ready;

  struct Candidate { double score; int f_opt; uword v; };
  static thread_local std::vector<Candidate> candidates;

  static thread_local std::vector<bool>                  committed;
  static thread_local std::vector<std::pair<int, uword>> commits;

  // Committed-delta storage for the conflict-resolution scan below: columns
  // of a pre-allocated D x n_opt matrix plus a parallel norm /
  // owning-endmember-index vector, instead of a vector of owned per-commit
  // vec objects. Recording a commit is a write into existing memory rather
  // than a fresh D-element heap allocation -- this loop can run many times
  // per cell (up to once per committed candidate, per pass).
  static thread_local mat committed_deltas_mat;
  static thread_local std::vector<double> committed_norms;
  static thread_local std::vector<int>    committed_ai;

  // Queued joint-pair retries for the current pass: candidates discarded for
  // conflicting with a structurally-collinear committed candidate. Cleared
  // and rebuilt every pass, since candidates themselves are regenerated
  // fresh each pass.
  struct QueuedRetry { int opt_i; uword v; };
  static thread_local std::vector<QueuedRetry> queued_retries;

  const uword n_active_sz = active_indices.size();

  resid.set_size(D);
  resid_raw.set_size(D);
  fluor_unmixed.set_size(F);
  cell_S.set_size(F + 1, D);
  cell_S_F_w.set_size(F, D);
  unmixed_full.set_size(F + 1);
  sqrt_w.set_size(D);
  dr.set_size(D);
  best_v.assign(n_opt, -1);
  init_f.set_size(F);
  base_resid_af.set_size(D);
  resid_w_af.set_size(D);
  cell_resid.set_size(D);
  y_hat.set_size(D);
  k_af_vec.set_size(nAF);
  cross_af.set_size(nAF);
  resid_sq_af.set_size(nAF);
  presid_af.set_size(nAF);
  pfluor_af.set_size(nAF);
  score_af_vec.set_size(nAF);
  w_af_init.set_size(F);
  cross_fluor_af.set_size(nAF);
  fluor_sq_af.set_size(nAF);
  coeff_init.set_size(F);
  other_unmixed.set_size(F > 0 ? F - 1 : 0);
  trial_unmixed.set_size(F);
  trial_resid_raw.set_size(D);
  trial_resid.set_size(D);
  S_F_w.set_size(F, D);

  A_base.set_size(F, F);
  b_base.set_size(F);
  y_vec.set_size(D);
  A_trial.set_size(F, F);
  b_trial.set_size(F);
  s_new.set_size(D);
  col_update.set_size(F);
  prev_row.set_size(D);

  rsw.set_size(D);
  w_eff.set_size(D);
  cross_v.set_size(max_n_variants);
  drsq_v.set_size(max_n_variants);
  g_cur.set_size(max_n_variants);

  a_base.set_size(F);
  c_vec.set_size(F > 0 ? F - 1 : 0);

  q_by_active.resize(n_active_sz);
  q_ready.assign(n_active_sz, 0);

  candidates.clear();
  candidates.reserve(4 * n_opt + 16);

  committed.reserve(n_opt);
  commits.reserve(n_opt);

  committed_deltas_mat.set_size(D, n_opt > 0 ? (uword)n_opt : 1);
  committed_norms.reserve(n_opt);
  committed_ai.reserve(n_opt);

  queued_retries.reserve(n_opt);

  auto score_af = [&](const vec& active_raw, uword& out_j, double& out_k) -> double {
    init_f        = P * active_raw;
    base_resid_af = active_raw - spectra.t() * init_f;

    // Every residual term below is taken in the same weighted inner product as
    // the abundance k, so presid_af measures the residual that k actually
    // minimises. r_dots_af_w is the unfloored weighted self-dot; the floor on
    // r_dots_af applies to the abundance denominator only.
    if (cell_weight) {
      resid_w_af = base_resid_af % w_global;
      k_af_vec   = clamp(r_lib_af_w.t() * active_raw, 0.0, arma::datum::inf) / r_dots_af;
      cross_af   = r_lib_af.t() * resid_w_af;
    } else {
      // Each r_j already lies in the orthogonal complement of the panel span,
      // so <r_j, Pperp y> == <r_j, y> and one gemv serves both the abundance
      // numerator and the residual cross-term.
      cross_af   = r_lib_af.t() * active_raw;
      k_af_vec   = clamp(cross_af, 0.0, arma::datum::inf) / r_dots_af;
      resid_w_af = base_resid_af;
    }

    const double base_resid_sq   = std::max(dot(resid_w_af, base_resid_af), 1e-16);
    const double base_resid_norm = std::sqrt(base_resid_sq);
    w_af_init = w_af % init_f;
    const double base_fluor_sq   = std::max(dot(w_af_init, w_af_init), 1e-16);
    const double base_fluor_norm = std::sqrt(base_fluor_sq);

    // Rank-1 residual-norm update for every candidate at once — avoids
    // forming a D-length r_j per candidate.
    resid_sq_af = base_resid_sq - 2.0 * (k_af_vec % cross_af)
      + (k_af_vec % k_af_vec % r_dots_af_w);
    presid_af   = sqrt(clamp(resid_sq_af, 0.0, arma::datum::inf)) / base_resid_norm;

    // Weighted-L2 fluorophore-leakage term for every candidate at once, via
    // the same rank-1 (quadratic) update as presid_af above.
    cross_fluor_af = v_lib_af_w.t() * w_af_init;
    fluor_sq_af    = base_fluor_sq - 2.0 * (k_af_vec % cross_fluor_af)
      + (k_af_vec % k_af_vec % c_fluor_af);
    pfluor_af      = sqrt(clamp(fluor_sq_af, 0.0, arma::datum::inf)) / base_fluor_norm;

    score_af_vec = presid_af % pfluor_af;

    const uword best_j = score_af_vec.index_min();
    out_j = best_j;
    out_k = k_af_vec[best_j];
    return score_af_vec[best_j];
  };

#pragma omp for schedule(dynamic, 64)
  for (uword i = 0; i < N; ++i) {
    const vec cell_raw = raw_data.unsafe_col(i);
    uword j_af; double k_af;
    score_af(cell_raw, j_af, k_af);
    af_index_vec[i]     = j_af;
    af_abundance_vec[i] = k_af;
    resid_mat.col(i)    = cell_raw - k_af * af_spectra_t.col(j_af);
  }

#pragma omp single
{
  if (n_af_passes > 1) {
    af_refine_cutoff = quantile_type7(af_abundance_vec, refine_af_quantile);
    for (uword i = 0; i < N; ++i)
      still_active[i] = (af_abundance_vec[i] >= af_refine_cutoff) ? 1u : 0u;
  }
}

for (int af_pass = 1; af_pass < n_af_passes; ++af_pass) {
#pragma omp for schedule(dynamic, 64)
  for (uword i = 0; i < N; ++i) {
    if (!still_active[i]) continue;
    uword j_ref; double k_ref;
    const double score_ref = score_af(resid_mat.col(i), j_ref, k_ref);
    if (score_ref < 1.0) {
      af_abundance_vec[i] += k_ref;
      resid_mat.col(i)    -= k_ref * af_spectra_t.col(j_ref);
    } else {
      still_active[i] = 0;
    }
  }
}

#pragma omp for schedule(dynamic, 64)
  for (uword i = 0; i < N; ++i) {

    const vec cell_raw   = raw_data.unsafe_col(i);
    cell_resid            = resid_mat.col(i);
    const double k_af     = af_abundance_vec[i];
    const uword best_j_af = af_index_vec[i];

    if (cell_weight) {
      S_F_w = spectra;
      S_F_w.each_row() %= sqrt_w_global.t();

      // Safe (non-throwing) form of solve(), with the returned success flag
      // actually checked: some cells' weighted detector-space systems are
      // numerically singular. The throwing single-return form used to sit
      // here -- an exception raised inside this OpenMP region is undefined
      // behaviour and crashes the host process with no message. On
      // failure, fall back to the same unweighted pseudo-inverse estimate
      // the cell_weight == FALSE branch below always uses, which cannot
      // itself fail (P is a fixed, precomputed matrix).
      if (!arma::solve(coeff_init, S_F_w.t(), cell_resid % sqrt_w_global,
                       arma::solve_opts::fast)) {
        coeff_init = P * cell_resid;
      }
      y_hat = (spectra.t() * coeff_init) + (cell_raw - cell_resid);
      for (uword d = 0; d < D; ++d)
        sqrt_w[d] = 1.0 / std::sqrt(std::max(std::abs(y_hat[d]), current_noise_floor[d]));

      cell_S_F_w = spectra.each_row() % sqrt_w.t();
      if (!arma::solve(fluor_unmixed, cell_S_F_w.t(), cell_resid % sqrt_w,
                       arma::solve_opts::fast)) {
        fluor_unmixed = P * cell_resid;
      }
    } else {
      // Detector weights are identical across cells here, so use global P
      sqrt_w.ones();
      cell_S_F_w    = spectra;
      fluor_unmixed = P * cell_resid;
    }

    resid_raw = cell_resid - spectra.t() * clamp(fluor_unmixed, 0.0, datum::inf);
    resid     = resid_raw % sqrt_w;

    // =====================================================================
    // B2. EARLY RETURN
    // =====================================================================
    if (af_only) {
      result(i, span(0, F - 1)) = fluor_unmixed.t();
      result(i, F)               = k_af;
      result(i, F + 1)           = (double)best_j_af + 1.0;
      continue;
    }

    // Only needed for the joint variant-selection pass loop and try_commit's
    // incremental Gram update below
    cell_S.rows(0, F - 1) = spectra;

    if (cell_weight) {
      A_base = cell_S_F_w * cell_S_F_w.t();
    } else {
      A_base = SST_global;   // constant across cells
    }
    b_base = cell_S_F_w * (cell_resid % sqrt_w);
    y_vec  = cell_resid % sqrt_w;

    const double cell_resid_ss = dot(cell_resid, cell_resid);
    std::fill(best_v.begin(), best_v.end(), -1);

    // Base-panel coefficients and the unconstrained residual sum of squares.
    // R0 comes straight out of the normal equations (|y|^2 - <a, S y>), so no
    // extra product is needed for it.
    a_base = fluor_unmixed;
    const double R0 = std::max(cell_resid_ss - dot(fluor_unmixed, b_base), 1e-12);


    // =====================================================================
    // C. JOINT VARIANT SELECTION
    // =====================================================================
    const int n_active = (int)active_indices.size();

    // Effective detector weighting for the residual-space scan.
    //   cell_weight == false -> sw is all ones, so the residual math collapses
    //   onto the statically precomputed r_lib / r_dots (no per-cell build).
    const vec& sw = cell_weight ? sqrt_w : sqrt_w_global;
    if (cell_weight) {
      w_eff = sw % sw;                               // per-cell detector weights
      std::fill(q_ready.begin(), q_ready.end(), 0);  // invalidate q cache
    }

    for (int pass = 0; pass < n_passes; ++pass) {

      const double rss_curr        = std::max(dot(resid, resid), 1e-12);
      const double rss_curr_sqrt   = std::sqrt(rss_curr);
      const double ratio_thresh_sq = 1.1025 * rss_curr;   // (1.05^2) * rss_curr
      double rss_accepted = dot(resid, resid);

      candidates.clear();
      queued_retries.clear();

      // resid weighted once per pass (independent of endmember).
      // <resid, x .* sw> == <resid .* sw, x>, so we fold sw into resid here.
      if (cell_weight) rsw = resid % sw; else rsw = resid;

      for (int ai = 0; ai < n_active; ++ai) {
        const int opt_i      = active_indices[ai];
        const FluorPrecomp& pc = precomp[opt_i];
        const double abund   = fluor_unmixed[pc.master_idx];
        if (abund < pos_thresholds[pc.master_idx]) continue;

        uword oi = 0;
        for (uword r = 0; r < F; ++r)
          if ((int)r != pc.master_idx) other_unmixed[oi++] = fluor_unmixed[r];

        const double base_leakage =
          std::max(dot(pc.w_leakage, abs(other_unmixed)), 1e-8);
        const int cur_v = best_v[opt_i];
        const uword nv  = pc.n_variants;

        if (use_exact) {
          // ---- Exact single-swap scan (Frisch-Waugh) ---------------------
          // With M the projector orthogonal to the other F-1 endmembers, the
          // panel re-solve after swapping variant v into this endmember has a
          // closed form:
          //   <M s_f, y> = a_f / c_ff                (base-panel coefficient)
          //   num_v      = <M s_f, y> + <M d_v, y>
          //   a_f^(v)    = num_v / q_v
          //   RSS^(v)    = (R0 + a_f^2/c_ff) - num_v^2 / q_v
          //   other^(v)  = (a_-f + a_f u_f) - a_f^(v) (u_f + h_v)
          // Every term but <M d_v, y> is precomputed or already in hand, so
          // the whole scan is one gemv plus scalar arithmetic, and the RSS is
          // exact rather than a fixed-abundance approximation.
          //
          // Ratios below are taken against the base panel (R0), not the
          // current residual, because the closed form is absolute rather than
          // incremental: it does not care which variant is currently applied.
          const double ms_y     = a_base[pc.master_idx] / pc.c_ff;
          const double R0_f     = R0 + a_base[pc.master_idx] * ms_y;
          const double R0_sqrt  = std::sqrt(R0);
          const double R0_gate  = 1.1025 * R0;            // (1.05^2) * R0

          cross_v.subvec(0, nv - 1) = pc.g_lib.t() * cell_resid;

          uword ob = 0;
          for (uword r = 0; r < F; ++r) {
            if ((int)r == pc.master_idx) continue;
            c_vec[ob] = a_base[r] + a_base[pc.master_idx] * pc.u_f[ob];
            ++ob;
          }

          for (uword v = 0; v < nv; ++v) {
            const double num_v   = ms_y + cross_v[v];
            const double new_rss = R0_f - num_v * num_v / pc.q[v];
            if (new_rss > R0_gate) continue;

            const double a_f_v = num_v / pc.q[v];

            double leak_num = 0.0;
            const double* uf = pc.u_f.memptr();
            const double* hv = pc.v_lib.colptr(v);
            for (uword o = 0; o < F - 1; ++o)
              leak_num += pc.w_leakage[o] *
                std::abs(c_vec[o] - a_f_v * (uf[o] + hv[o]));
            const double leakage_ratio = leak_num / base_leakage;

            const double resid_ratio =
              std::sqrt(std::max(new_rss, 0.0)) / R0_sqrt;
            const double joint_score =
              std::pow(std::max(resid_ratio,   1e-8), alpha) *
              std::pow(std::max(leakage_ratio, 1e-8), 1.0 - alpha);

            if (joint_score < 1.0)
              candidates.push_back({joint_score, ai, v});
          }
        } else {
          // ---- Vectorised residual-ratio scan over all variants at once ----
          // cross_v[v] = <resid, (r_v - r_cur) .* w>      (1 gemv, shared rsw)
          // drsq_v[v]  = ||(r_v - r_cur) .* w||^2         (hoisted q + 1 gemv)
          // Weighted self-dots q: built once per active endmember per cell (after
          // the skip check, so skipped endmembers cost nothing) and reused across
          // passes. Unweighted uses the static precomputed r_dots.
          //
          // Writes below go into the leading nv elements of the pre-sized
          // cross_v / drsq_v / g_cur buffers -- endmembers with fewer variants
          // than max_n_variants never trigger a resize.
          if (cell_weight && !q_ready[ai]) {
            q_by_active[ai] = pc.r_lib_sq.t() * w_eff;
            q_ready[ai] = 1;
          }
          const vec& q_ref = cell_weight ? q_by_active[ai] : pc.r_dots;
          cross_v.subvec(0, nv - 1) = pc.r_lib.t() * rsw;         // n_variants
          if (cur_v < 0) {
            drsq_v.subvec(0, nv - 1) = q_ref;
          } else {
            if (cell_weight) g_cur.subvec(0, nv - 1) = pc.r_lib.t() * (pc.r_lib.col(cur_v) % w_eff);
            else             g_cur.subvec(0, nv - 1) = pc.r_lib.t() *  pc.r_lib.col(cur_v);
            drsq_v.subvec(0, nv - 1) = q_ref + q_ref[cur_v] - 2.0 * g_cur.subvec(0, nv - 1);
            cross_v.subvec(0, nv - 1) -= cross_v[cur_v];
          }

          const double abund2 = abund * abund;
          for (uword v = 0; v < pc.n_variants; ++v) {
            const double new_rss =
              rss_curr - 2.0 * abund * cross_v[v] + abund2 * drsq_v[v];
            if (new_rss > ratio_thresh_sq) continue;   // == resid_ratio > 1.05

            // Leakage penalty — only for residual-passing variants. Scalar loop
            // over the F-1 "other" endmembers, no vector temporaries.
            //   new_other[o] = other_unmixed[o] - abund*(v_lib[o,v] - v_lib[o,cur])
            double leak_num = 0.0;
            const double* vl = pc.v_lib.colptr(v);
            if (cur_v < 0) {
              for (uword o = 0; o < F - 1; ++o)
                leak_num += pc.w_leakage[o] *
                  std::abs(other_unmixed[o] - abund * vl[o]);
            } else {
              const double* vlc = pc.v_lib.colptr(cur_v);
              for (uword o = 0; o < F - 1; ++o)
                leak_num += pc.w_leakage[o] *
                  std::abs(other_unmixed[o] - abund * (vl[o] - vlc[o]));
            }
            const double leakage_ratio = leak_num / base_leakage;

            const double resid_ratio =
              std::sqrt(std::max(new_rss, 0.0)) / rss_curr_sqrt;
            const double joint_score =
              std::pow(std::max(resid_ratio,   1e-8), alpha) *
              std::pow(std::max(leakage_ratio, 1e-8), 1.0 - alpha);

            if (joint_score < 1.0)
              candidates.push_back({joint_score, ai, v});
          }
        }
      }

      if (candidates.empty()) break;

      // Sort best-first
      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b){
                  return a.score < b.score;
                });

      // Conflict resolution: commit non-overlapping swaps
      committed.assign(n_active, false);
      uword n_committed = 0;
      committed_norms.clear();
      committed_ai.clear();
      commits.clear();

      for (const auto& cand : candidates) {
        if (committed[cand.f_opt]) continue;

        const int opt_i        = active_indices[cand.f_opt];
        const FluorPrecomp& pc = precomp[opt_i];
        const double abund     = fluor_unmixed[pc.master_idx];
        const int cur_v        = best_v[opt_i];

        if (cur_v < 0) {
          dr = pc.r_lib.col(cand.v) * abund;
        } else {
          dr = (pc.r_lib.col(cand.v) - pc.r_lib.col(cur_v)) * abund;
        }
        const double dr_norm = std::max(norm(dr), 1e-12);

        bool conflict = false;
        for (uword c = 0; c < n_committed; ++c) {
          const double cosine = std::abs(dot(dr, committed_deltas_mat.col(c))) /
            (dr_norm * committed_norms[c]);
          if (cosine > 0.5) {
            conflict = true;
            const int winner_master = precomp[active_indices[committed_ai[c]]].master_idx;
            const bool collinear_pair = (bool)is_collinear(pc.master_idx, winner_master);

            if (joint_pair_resolution && collinear_pair) {
              queued_retries.push_back({opt_i, cand.v});
            }
            break;
          }
        }
        if (conflict) continue;

        committed[cand.f_opt] = true;
        committed_deltas_mat.col(n_committed) = dr;
        committed_norms.push_back(dr_norm);
        committed_ai.push_back(cand.f_opt);
        ++n_committed;
        commits.push_back({opt_i, cand.v});
      }

      if (commits.empty()) break;

      auto try_commit = [&](int opt_i, uword v) -> bool {
        const FluorPrecomp& pc_v = precomp[opt_i];
        const int idx = pc_v.master_idx;

        prev_row = cell_S.row(idx);
        cell_S.row(idx) = pc_v.v_mats.row(v);

        s_new      = pc_v.v_mats.row(v) % sqrt_w.t();
        col_update = cell_S_F_w * s_new.t();

        A_trial = A_base;
        for (uword r = 0; r < F; ++r) {
          A_trial(r, idx) = col_update[r];
          A_trial(idx, r) = col_update[r];
        }
        A_trial(idx, idx) = arma::dot(s_new, s_new);

        b_trial = b_base;
        b_trial[idx] = arma::dot(s_new, y_vec);

        // A_trial is a Gram matrix (symmetric positive-definite); use the
        // Cholesky path, falling back to LU automatically if not SPD.
        bool success = arma::solve(trial_unmixed, A_trial, b_trial,
                                   arma::solve_opts::fast + arma::solve_opts::likely_sympd);
        if (!success) {
          cell_S.row(idx) = prev_row; // Linear dependency fault protection
          return false;
        }

        trial_resid_raw = cell_resid - cell_S.rows(0, F - 1).t() * clamp(trial_unmixed, 0.0, datum::inf);
        trial_resid = trial_resid_raw % sqrt_w;
        const double trial_rss = dot(trial_resid, trial_resid);

        if (trial_rss < rss_accepted) {
          best_v[opt_i] = (int)v;
          fluor_unmixed = trial_unmixed;
          resid_raw     = trial_resid_raw;
          resid         = trial_resid;
          rss_accepted  = trial_rss;

          cell_S_F_w.row(idx) = s_new;
          A_base = A_trial;
          b_base = b_trial;
          return true;
        } else {
          cell_S.row(idx) = prev_row;
          return false;
        }
      };

      for (auto& [opt_i, v] : commits) try_commit(opt_i, v);

      // Joint-pair retry: replay candidates discarded for conflicting with a
      // structurally-collinear committed candidate, now that partner's own
      // swap (if any) has already been applied above.
      if (joint_pair_resolution) {
        for (const auto& qr : queued_retries)
          try_commit(qr.opt_i, qr.v);
      }

      if (dot(resid_raw, resid_raw) < 1e-16 * cell_resid_ss) break;
    } // end for pass

    // =====================================================================
    // D. Write output
    // =====================================================================
    result(i, span(0, F - 1)) = fluor_unmixed.t();
    result(i, F)               = k_af;
    result(i, F + 1)           = (double)best_j_af + 1.0;
  } // end for i
} // end pragma omp parallel

return result;
}
