// fix_my_unmix_kernel.cpp
// -----------------------
// Batched anchored spillover slopes for Fix My Unmix (AutoSpectral tab,
// Spectral Correction page), built as a cffi extension by
// build_fix_my_unmix_kernel.py. It is the compiled form of anchored_slopes()
// in controller_components/fix_my_unmix.py and gives the same numbers.
//
// One source abundance is shared by m targets. For each target:
//   - the source-negative bulk events that are negative for the target, once
//     the current slope is removed, give the anchor: their median source and
//     target values;
//   - a Huber line through the anchor is fitted to the clearly positive
//     source events that are negative for the target (upper boundary), and
//     after the first pass also above the target's lower boundary (band);
//   - negativity is re-decided with the new slope up to max_mask_passes
//     times, stopping once the slope moves by less than mask_tolerance of
//     itself or leaves max_coefficient.
// The caller selects and subsamples the positive and bulk events, so the
// kernel draws no random numbers. Targets are independent; worker threads
// (std::thread, no OpenMP) take them one at a time from a shared counter.
//
// Layout: the per-target arrays are target-major (m x n, C order).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

// Median of src[0..n) through a reused scratch buffer (numpy's convention for even n).
static double median_into(std::vector<double>& scratch, const double* src, int n) {
    std::copy(src, src + n, scratch.begin());
    std::size_t half = static_cast<std::size_t>(n) / 2;
    std::nth_element(scratch.begin(), scratch.begin() + half, scratch.begin() + n);
    if (n % 2 == 1) return scratch[half];
    double upper = scratch[half];
    double lower = *std::max_element(scratch.begin(), scratch.begin() + half);
    return (lower + upper) / 2.0;
}

static double robust_scale(const std::vector<double>& r, int n, std::vector<double>& absdev,
                           std::vector<double>& scratch) {
    const double meps = std::numeric_limits<double>::epsilon();
    double med = median_into(scratch, r.data(), n);
    for (int i = 0; i < n; ++i) absdev[i] = std::fabs(r[i] - med);
    return std::max(median_into(scratch, absdev.data(), n) / 0.6745, meps);
}

// Huber IRLS slope of a line through the origin; returns false (slope 0, se
// NaN) when it does not converge.
static bool huber_origin(const double* xs, const double* ys, int n, double start, double k, int max_iter,
                         double tol, double& slope, double& se, std::vector<double>& r,
                         std::vector<double>& absdev, std::vector<double>& scratch) {
    const double meps = std::numeric_limits<double>::epsilon();
    double b = start;
    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < n; ++i) r[i] = ys[i] - b * xs[i];
        double scale = robust_scale(r, n, absdev, scratch);
        double sxx = 0.0, sxy = 0.0;
        for (int i = 0; i < n; ++i) {
            double w = std::min(1.0, k / std::max(std::fabs(r[i] / scale), meps));
            sxx += w * xs[i] * xs[i];
            sxy += w * xs[i] * ys[i];
        }
        if (!(sxx > 0.0)) break;
        double b_new = sxy / sxx;
        bool settled = std::fabs(b_new - b) < tol * std::max(std::fabs(b_new), 1.0);
        b = b_new;
        if (settled) {
            for (int i = 0; i < n; ++i) r[i] = ys[i] - b * xs[i];
            double s2 = robust_scale(r, n, absdev, scratch);
            double wxx = 0.0;
            for (int i = 0; i < n; ++i) {
                double w = std::min(1.0, k / std::max(std::fabs(r[i] / s2), meps));
                wxx += w * xs[i] * xs[i];
            }
            slope = b;
            se = s2 / std::sqrt(std::max(wxx, meps));
            return true;
        }
    }
    slope = 0.0;
    se = std::numeric_limits<double>::quiet_NaN();
    return false;
}

namespace {

struct Batch {
    const double* xb; const double* Yb; const double* Tb; const double* Lb; int nb;
    const double* xk; const double* Yk; const double* Tk; int nk;
    const double* start; int m;
    double max_coefficient; int max_mask_passes; double mask_tolerance; int min_count; int band;
    double k; int max_iter; double tol;
    double* slope_out; int* n_out; double* se_out;
};

void run_targets(const Batch& b, std::atomic<int>& next) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    int cap = std::max(b.nb, b.nk);
    std::vector<double> xs(cap), ys(cap), r(cap), absdev(cap), scratch(cap);
    for (int j = next.fetch_add(1); j < b.m; j = next.fetch_add(1)) {
        const double* yb = b.Yb + static_cast<std::size_t>(j) * b.nb;
        const double* tb = b.Tb + static_cast<std::size_t>(j) * b.nb;
        const double* lb = b.Lb ? b.Lb + static_cast<std::size_t>(j) * b.nb : nullptr;
        const double* yk = b.Yk + static_cast<std::size_t>(j) * b.nk;
        const double* tk = b.Tk + static_cast<std::size_t>(j) * b.nk;
        double slope_curr = b.start[j];
        if (!std::isfinite(slope_curr) || std::fabs(slope_curr) > b.max_coefficient) slope_curr = 0.0;
        double slope_res = nan, se_res = nan;
        int n_res = 0;
        for (int pass = 0; pass <= b.max_mask_passes; ++pass) {
            int nkn = 0;
            for (int i = 0; i < b.nk; ++i) {
                if (yk[i] - slope_curr * b.xk[i] < tk[i]) { xs[nkn] = b.xk[i]; ys[nkn] = yk[i]; ++nkn; }
            }
            int nbn = 0;
            for (int i = 0; i < b.nb; ++i) {
                double resid = yb[i] - slope_curr * b.xb[i];
                bool keep = resid < tb[i];
                if (keep && b.band && lb && pass > 0) keep = resid > lb[i];
                if (keep) ++nbn;
            }
            if (nkn < b.min_count || nbn < b.min_count) break;
            double x0 = median_into(scratch, xs.data(), nkn);
            double y0 = median_into(scratch, ys.data(), nkn);
            int t = 0;
            for (int i = 0; i < b.nb; ++i) {
                double resid = yb[i] - slope_curr * b.xb[i];
                bool keep = resid < tb[i];
                if (keep && b.band && lb && pass > 0) keep = resid > lb[i];
                if (keep) { xs[t] = b.xb[i] - x0; ys[t] = yb[i] - y0; ++t; }
            }
            double slope_next = 0.0, se = nan;
            huber_origin(xs.data(), ys.data(), t, slope_curr, b.k, b.max_iter, b.tol, slope_next, se, r, absdev,
                         scratch);
            n_res = t;
            slope_res = slope_next;
            se_res = se;
            bool valid = std::fabs(slope_next) <= b.max_coefficient;
            bool settled = std::fabs(slope_next - slope_curr) <= b.mask_tolerance * std::fabs(slope_next);
            slope_curr = slope_next;
            if (!valid || settled) break;
        }
        b.slope_out[j] = slope_res;
        b.n_out[j] = n_res;
        b.se_out[j] = se_res;
    }
}

}  // namespace

extern "C" {

// Hardware threads available (at least 1).
int fmu_max_threads(void) {
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

// slope_out[j] / se_out[j] are NaN where fewer than min_count positive or
// bulk events were negative for the target on the first pass; n_out[j] is
// the last pass's positive event count. Lb may be NULL (no band).
// n_threads 0 uses every hardware thread.
void fmu_anchored_batch(const double* xb, const double* Yb, const double* Tb, const double* Lb, int nb,
                        const double* xk, const double* Yk, const double* Tk, int nk,
                        const double* start, int m, double max_coefficient, int max_mask_passes,
                        double mask_tolerance, int min_count, int band, double k, int max_iter, double tol,
                        int n_threads, double* slope_out, int* n_out, double* se_out) {
    Batch b{xb, Yb, Tb, Lb, nb, xk, Yk, Tk, nk, start, m, max_coefficient, max_mask_passes, mask_tolerance,
            min_count, band, k, max_iter, tol, slope_out, n_out, se_out};
    int workers = n_threads > 0 ? n_threads : fmu_max_threads();
    workers = std::max(1, std::min(workers, m));
    std::atomic<int> next(0);
    if (workers == 1) {
        run_targets(b, next);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (int t = 1; t < workers; ++t) pool.emplace_back(run_targets, std::cref(b), std::ref(next));
    run_targets(b, next);
    for (auto& th : pool) th.join();
}

}  // extern "C"
