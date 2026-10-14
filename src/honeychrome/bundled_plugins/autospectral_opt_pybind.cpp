// autospectral_opt_pybind.cpp
// ---------------------------------------------------------------------------
// pybind11 binding over unmix_autospectral_joint_core() (core_joint_unmix.hpp).
// Converts numpy arrays <-> Armadillo types.
// ---------------------------------------------------------------------------
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <armadillo>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core_joint_unmix.hpp"

namespace py = pybind11;

namespace {

using ContigArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// numpy is row-major (C order); Armadillo is column-major (Fortran order).
// Reading the same memory as (n_cols, n_rows) column-major and transposing
// gives the correct (n_rows, n_cols) matrix. copy_aux_mem=true so the
// returned arma::mat owns its memory once the numpy array goes out of scope.
arma::mat np_to_arma_mat(const ContigArray& arr) {
  auto buf = arr.request();
  if (buf.ndim != 2)
    throw std::invalid_argument("expected a 2-D array");
  const arma::uword n_rows = static_cast<arma::uword>(buf.shape[0]);
  const arma::uword n_cols = static_cast<arma::uword>(buf.shape[1]);
  arma::mat tmp(static_cast<double*>(buf.ptr), n_cols, n_rows, /*copy_aux_mem=*/true);
  return tmp.t();
}

arma::vec np_to_arma_vec(const ContigArray& arr) {
  auto buf = arr.request();
  if (buf.ndim != 1)
    throw std::invalid_argument("expected a 1-D array");
  const arma::uword n = static_cast<arma::uword>(buf.shape[0]);
  return arma::vec(static_cast<double*>(buf.ptr), n, /*copy_aux_mem=*/true);
}

py::array_t<double> arma_to_np(const arma::mat& m) {
  py::array_t<double> out({static_cast<py::ssize_t>(m.n_rows),
                            static_cast<py::ssize_t>(m.n_cols)});
  auto buf = out.request();
  double* dst = static_cast<double*>(buf.ptr);
  const arma::mat mt = m.t();   // column-major (n_cols,n_rows) memory == row-major (n_rows,n_cols)
  std::memcpy(dst, mt.memptr(), sizeof(double) * m.n_rows * m.n_cols);
  return out;
}

// None -> joint L2 scoring; otherwise a dict with null_var ((3 * n_mean_rows)
// x F: c0, c1, c2 of each mean row in turn), spread (None or an F x F array)
// and mean_rows (list of AF library rows).
AfEnvelopeInput envelope_from_dict(const py::dict& d) {
  AfEnvelopeInput env;
  env.null_var = np_to_arma_mat(d["null_var"].cast<ContigArray>()).t();
  if (d.contains("spread") && !d["spread"].is_none())
    env.spread = np_to_arma_mat(d["spread"].cast<ContigArray>());
  if (d.contains("mean_rows") && !d["mean_rows"].is_none()) {
    const auto rows = d["mean_rows"].cast<std::vector<long long>>();
    env.mean_rows.set_size(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i] < 0) throw std::invalid_argument("envelope mean_rows must be non-negative.");
      env.mean_rows[i] = static_cast<arma::uword>(rows[i]);
    }
  }
  return env;
}

}  // namespace

static py::array_t<double> unmix_autospectral_joint(
    ContigArray                 raw_data_in,
    ContigArray                 spectra,
    ContigArray                 af_spectra,
    const std::vector<std::string>& fluor_names,
    ContigArray                 pos_thresholds,
    const py::list&             variants,   // list of dict(name, v_mats, delta_obs)
    int                         n_passes,
    int                         n_threads,
    bool                        cell_weight,
    py::object                  noise_floor,   // None or 1-D ndarray
    double                      alpha,
    double                      collinear_thresh,
    bool                        joint_pair_resolution,
    int                         n_af_passes,
    double                      refine_af_quantile,
    bool                        exact_variant_scan,
    py::object                  envelope       // None or dict, see envelope_from_dict
) {
  std::vector<FluorVariantInput> cpp_variants;
  cpp_variants.reserve(variants.size());
  for (const auto& item : variants) {
    py::dict d = item.cast<py::dict>();
    FluorVariantInput fv;
    fv.name      = d["name"].cast<std::string>();
    fv.v_mats    = np_to_arma_mat(d["v_mats"].cast<ContigArray>());
    fv.delta_obs = np_to_arma_mat(d["delta_obs"].cast<ContigArray>());
    cpp_variants.push_back(std::move(fv));
  }

  AfEnvelopeInput env;
  const bool has_envelope = !envelope.is_none();
  if (has_envelope)
    env = envelope_from_dict(envelope.cast<py::dict>());

  arma::vec nf;   // stays empty if noise_floor is None -> core fills default
  if (!noise_floor.is_none())
    nf = np_to_arma_vec(noise_floor.cast<ContigArray>());

  // Inputs are copied into Armadillo types before the GIL is released.
  const arma::mat raw_mat     = np_to_arma_mat(raw_data_in);
  const arma::mat spectra_mat = np_to_arma_mat(spectra);
  const arma::mat af_mat      = np_to_arma_mat(af_spectra);
  const arma::vec pos_vec     = np_to_arma_vec(pos_thresholds);

  arma::mat result;
  {
    py::gil_scoped_release release;   // lets the Qt main thread run during the solve
    result = unmix_autospectral_joint_core(
        raw_mat, spectra_mat, af_mat, fluor_names, pos_vec, cpp_variants,
        n_passes, n_threads, cell_weight,
        &nf,
        alpha, collinear_thresh, joint_pair_resolution,
        n_af_passes, refine_af_quantile, exact_variant_scan,
        has_envelope ? &env : nullptr
    );
  }

  return arma_to_np(result);
}

PYBIND11_MODULE(_autospectral_opt_kernel, m) {
  m.doc() = "Honeychrome AutoSpectral Optimization joint AF + variant "
            "unmixing kernel (pybind11 / Armadillo).";
  m.def("unmix_autospectral_joint", &unmix_autospectral_joint,
        py::arg("raw_data_in"), py::arg("spectra"), py::arg("af_spectra"),
        py::arg("fluor_names"), py::arg("pos_thresholds"), py::arg("variants"),
        py::arg("n_passes") = 1, py::arg("n_threads") = 1,
        py::arg("cell_weight") = false, py::arg("noise_floor") = py::none(),
        py::arg("alpha") = 0.5, py::arg("collinear_thresh") = 0.5,
        py::arg("joint_pair_resolution") = true, py::arg("n_af_passes") = 1,
        py::arg("refine_af_quantile") = 0.5, py::arg("exact_variant_scan") = false,
        py::arg("envelope") = py::none(),
        "Joint per-cell AF + fluorophore-variant unmixing. Returns an "
        "(N, F+2) array: [fluor abundances | AF abundance | AF index (1-based)]. "
        "envelope (dict of null_var, spread, mean_rows) scores AF candidates "
        "by the positivity-weighted term; None scores them by joint L2.");
  m.attr("SUPPORTS_ENVELOPE") = true;              // unmix_autospectral_joint takes an envelope
  m.attr("SUPPORTS_POSITIVITY_ENVELOPE") = true;   // and scores it with the positivity-weighted term
}
