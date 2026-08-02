/*
 * som_kernel.c
 * ------------
 * OpenMP batch Self-Organizing Map training + nearest-code assignment,
 * for the FlowSOM clustering path of the DR/Clustering plugin
 * (flowsom_consensus.py).
 *
 * Ported from AutoSpectral's som_train_batch_cpp() / map_data_to_codes_cpp()
 * (Rcpp), themselves adapted from EmbedSOM's bsom() -- see project history
 * (som.cpp, embedsom_som.cpp). Batch SOM: one full-dataset pass (epoch)
 * between codebook updates -- not pyFlowSOM's retired online/per-event
 * trainer. See flowsom_consensus.py module docstring for the rationale.
 *
 * All 2-D arrays are C-contiguous (row-major), matching af_kernel.c's
 * convention: element (row, col) of an (nrow, ncol) array is
 * ptr[row*ncol + col].
 *
 * Parallelisation strategy (same as som.cpp):
 *   - Assignment (Step 1) is parallelised over events, one accumulator
 *     buffer PER THREAD (heap arrays, indexed by omp_get_thread_num()),
 *     reduced serially afterward -- cheap relative to the assignment
 *     step itself.
 *   - Neighbourhood diffusion update (Step 2) is parallelised over
 *     destination node: each iteration only reads the already-reduced
 *     global sums and writes its own distinct row of the codebook.
 *
 * Compile: see build_som_kernel.py (cffi, mirrors build_af_kernel.py).
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- distance functions ------------------------------------------------
 * p1, p2: row pointers (px contiguous doubles each) -- already offset by
 * the caller, so no stride/leading-dimension argument needed (unlike the
 * Fortran-order version in som.cpp).
 */

static double dist_eucl(const double *p1, const double *p2, int px) {
    double s = 0.0;
    for (int j = 0; j < px; j++) { double d = p1[j] - p2[j]; s += d * d; }
    return sqrt(s);
}

static double dist_manh(const double *p1, const double *p2, int px) {
    double s = 0.0;
    for (int j = 0; j < px; j++) s += fabs(p1[j] - p2[j]);
    return s;
}

static double dist_chebyshev(const double *p1, const double *p2, int px) {
    double s = 0.0;
    for (int j = 0; j < px; j++) { double d = fabs(p1[j] - p2[j]); if (d > s) s = d; }
    return s;
}

static double dist_cosine(const double *p1, const double *p2, int px) {
    double nom = 0.0, d1 = 0.0, d2 = 0.0;
    for (int j = 0; j < px; j++) { nom += p1[j] * p2[j]; d1 += p1[j] * p1[j]; d2 += p2[j] * p2[j]; }
    return (-nom / (sqrt(d1) * sqrt(d2))) + 1.0;
}

typedef double (*dist_fun)(const double *, const double *, int);

static dist_fun get_dist_fun(int dist) {
    switch (dist) {
        case 1:  return dist_manh;
        case 3:  return dist_chebyshev;
        case 4:  return dist_cosine;
        case 2:
        default: return dist_eucl;
    }
}

/* ---- training -----------------------------------------------------------
 *
 * data       : (n, px)          training events
 * init_codes : (ncodes, px)     initial codebook -- caller picks (e.g. a
 *              random sample of data rows), so seeding stays in Python
 * nhbrdist   : (ncodes, ncodes) grid neighbourhood distances (Chebyshev)
 * radii      : (n_epochs,)      neighbourhood radius per epoch -- must
 *              stay strictly positive (Gaussian kernel exp(-d^2/radius^2));
 *              guarded internally via min_radius but the caller should not
 *              rely on that guard (build the schedule with a positive floor)
 * codes_out  : (ncodes, px)     [OUT] trained codebook
 */
void som_train_batch(
    const double *data,
    const double *init_codes,
    const double *nhbrdist,
    const double *radii,
    double       *codes_out,
    int n, int px, int ncodes, int n_epochs,
    int dist, int n_threads
)
{
    dist_fun distf = get_dist_fun(dist);

    #ifdef _OPENMP
    if (n_threads < 1) n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);
    #else
    n_threads = 1;
    #endif

    memcpy(codes_out, init_codes, (size_t)ncodes * px * sizeof(double));

    double *thread_sums   = calloc((size_t)n_threads * ncodes * px, sizeof(double));
    double *thread_counts = calloc((size_t)n_threads * ncodes,      sizeof(double));
    double *global_sums   = malloc((size_t)ncodes * px * sizeof(double));
    double *global_counts = malloc((size_t)ncodes * sizeof(double));
    double *prev_codes    = malloc((size_t)ncodes * px * sizeof(double));

    const double min_radius = 1e-10;

    for (int epoch = 0; epoch < n_epochs; epoch++) {

        memset(thread_sums,   0, (size_t)n_threads * ncodes * px * sizeof(double));
        memset(thread_counts, 0, (size_t)n_threads * ncodes * sizeof(double));

        /* --- Step 1: assign every event to its nearest code (parallel over events) --- */
        #pragma omp parallel default(shared)
        {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            double *my_sums   = thread_sums   + (size_t)tid * ncodes * px;
            double *my_counts = thread_counts + (size_t)tid * ncodes;

            #pragma omp for schedule(static)
            for (int ev = 0; ev < n; ev++) {
                const double *p = data + (size_t)ev * px;
                int nearest = 0;
                double neard = distf(p, codes_out, px);
                for (int cd = 1; cd < ncodes; cd++) {
                    double d = distf(p, codes_out + (size_t)cd * px, px);
                    if (d < neard) { neard = d; nearest = cd; }
                }
                my_counts[nearest] += 1.0;
                for (int j = 0; j < px; j++)
                    my_sums[(size_t)nearest * px + j] += p[j];
            }
        } /* end omp parallel -- barrier guarantees every thread's buffer is complete */

        /* --- reduce per-thread buffers (serial, O(threads * ncodes * px)) --- */
        memset(global_sums,   0, (size_t)ncodes * px * sizeof(double));
        memset(global_counts, 0, (size_t)ncodes * sizeof(double));
        for (int t = 0; t < n_threads; t++) {
            const double *ts = thread_sums   + (size_t)t * ncodes * px;
            const double *tc = thread_counts + (size_t)t * ncodes;
            for (size_t idx = 0; idx < (size_t)ncodes * px; idx++) global_sums[idx] += ts[idx];
            for (int cd = 0; cd < ncodes; cd++)                    global_counts[cd] += tc[cd];
        }

        memcpy(prev_codes, codes_out, (size_t)ncodes * px * sizeof(double));

        /* --- Step 2: neighbourhood-weighted diffusion update (parallel over destination node) --- */
        double radius = radii[epoch];
        if (radius < min_radius) radius = min_radius;
        double inv_sq_radius = -1.0 / (radius * radius);

        #pragma omp parallel for schedule(static)
        for (int di = 0; di < ncodes; di++) {
            double *out_row = codes_out + (size_t)di * px;
            double *row_sum = malloc((size_t)px * sizeof(double));
            for (int j = 0; j < px; j++) row_sum[j] = 0.0;
            double row_weight = 0.0;

            for (int si = 0; si < ncodes; si++) {
                double d = nhbrdist[(size_t)di * ncodes + si];
                double w = exp(d * d * inv_sq_radius);
                row_weight += w * global_counts[si];
                const double *gs = global_sums + (size_t)si * px;
                for (int j = 0; j < px; j++) row_sum[j] += gs[j] * w;
            }

            if (row_weight > 0.0) {
                for (int j = 0; j < px; j++) out_row[j] = row_sum[j] / row_weight;
            } else {
                /* No node contributed within kernel range -- hold previous
                 * position rather than collapsing to zero. */
                memcpy(out_row, prev_codes + (size_t)di * px, (size_t)px * sizeof(double));
            }
            free(row_sum);
        }
    }

    free(thread_sums);
    free(thread_counts);
    free(global_sums);
    free(global_counts);
    free(prev_codes);
}

/* ---- mapping --------------------------------------------------------------
 * Fully data-parallel across events -- no shared mutable state, no barrier.
 *
 * data        : (n, px)      events to map
 * codes       : (ncodes, px) trained codebook
 * out_node_id : (n,) [OUT]   0-based nearest-code index per event
 * out_dist    : (n,) [OUT]   distance to that code
 */
void som_map_to_codes(
    const double *data,
    const double *codes,
    int32_t      *out_node_id,
    double       *out_dist,
    int n, int px, int ncodes,
    int dist, int n_threads
)
{
    dist_fun distf = get_dist_fun(dist);

    #ifdef _OPENMP
    if (n_threads < 1) n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *p = data + (size_t)i * px;
        int nearest = 0;
        double neard = distf(p, codes, px);
        for (int cd = 1; cd < ncodes; cd++) {
            double d = distf(p, codes + (size_t)cd * px, px);
            if (d < neard) { neard = d; nearest = cd; }
        }
        out_node_id[i] = nearest;
        out_dist[i]    = neard;
    }
}