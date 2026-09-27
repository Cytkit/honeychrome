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
    bool                        exact_variant_scan
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

  arma::vec nf;   // stays empty if noise_floor is None -> core fills default
  if (!noise_floor.is_none())
    nf = np_to_arma_vec(noise_floor.cast<ContigArray>());

  arma::mat result = unmix_autospectral_joint_core(
      np_to_arma_mat(raw_data_in),
      np_to_arma_mat(spectra),
      np_to_arma_mat(af_spectra),
      fluor_names,
      np_to_arma_vec(pos_thresholds),
      cpp_variants,
      n_passes, n_threads, cell_weight,
      &nf,
      alpha, collinear_thresh, joint_pair_resolution,
      n_af_passes, refine_af_quantile, exact_variant_scan
  );

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
        "Joint per-cell AF + fluorophore-variant unmixing. Returns an "
        "(N, F+2) array: [fluor abundances | AF abundance | AF index (1-based)].");
}
