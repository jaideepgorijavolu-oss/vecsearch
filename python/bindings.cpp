// pybind11 bindings: vecsearch._core
//
// Input vectors are read in place from the NumPy buffer (zero copy). We do not let pybind11
// convert arguments: a float64 or non-contiguous array raises an error that says what to do,
// instead of silently copying a large array. The GIL is released while C++ runs, so Python
// threads (e.g. the web service) can search in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>

#include "vecsearch/distance.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"
#include "vecsearch/version.hpp"

namespace py = pybind11;
using namespace vecsearch;

namespace {

// A validated view of a (n, dim) float32 C-contiguous array. Holds a reference so the buffer
// stays alive while the GIL is released.
struct Matrix {
  py::array array;
  const float* data;
  std::size_t rows;
};

Matrix as_matrix(const py::handle& obj, std::size_t dim, const char* what) {
  if (!py::isinstance<py::array>(obj)) {
    throw py::type_error(std::string(what) + " must be a numpy.ndarray of float32, got " +
                         std::string(py::str(py::type::of(obj).attr("__name__"))));
  }
  auto arr = py::reinterpret_borrow<py::array>(obj);
  if (!arr.dtype().is(py::dtype::of<float>())) {
    throw py::type_error(std::string(what) + " must have dtype float32, got " +
                         std::string(py::str(arr.dtype())) + " (use arr.astype(np.float32))");
  }
  if (!(arr.flags() & py::array::c_style)) {
    throw py::value_error(std::string(what) +
                          " must be C-contiguous (use np.ascontiguousarray(arr))");
  }
  std::size_t rows;
  if (arr.ndim() == 1) {
    rows = 1;
    if (static_cast<std::size_t>(arr.shape(0)) != dim) {
      throw py::value_error(std::string(what) + " has dimension " + std::to_string(arr.shape(0)) +
                            ", index has dimension " + std::to_string(dim));
    }
  } else if (arr.ndim() == 2) {
    rows = static_cast<std::size_t>(arr.shape(0));
    if (static_cast<std::size_t>(arr.shape(1)) != dim) {
      throw py::value_error(std::string(what) + " has dimension " + std::to_string(arr.shape(1)) +
                            ", index has dimension " + std::to_string(dim));
    }
  } else {
    throw py::value_error(std::string(what) + " must be 1-D or 2-D, got " +
                          std::to_string(arr.ndim()) + "-D");
  }
  return {arr, static_cast<const float*>(arr.data()), rows};
}

using Labels = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// Labels are small next to the vectors, so any integer array (or list) is accepted and
// converted to int64.
Labels as_labels(const py::handle& obj, const char* what) {
  Labels labels = Labels::ensure(obj);
  if (!labels) throw py::type_error(std::string(what) + " must be an array of integers");
  if (labels.ndim() != 1) throw py::value_error(std::string(what) + " must be 1-D");
  return labels;
}

// Hands a SearchResult's buffers to NumPy without copying: the arrays own the result through
// a capsule.
py::tuple to_numpy(SearchResult&& r) {
  auto* owner = new SearchResult(std::move(r));
  py::capsule free_when_done(owner, [](void* p) { delete static_cast<SearchResult*>(p); });
  const auto nq = static_cast<py::ssize_t>(owner->num_queries);
  const auto k = static_cast<py::ssize_t>(owner->k);
  py::array_t<std::int64_t> ids({nq, k}, owner->ids.data(), free_when_done);
  py::array_t<float> dists({nq, k}, owner->distances.data(), free_when_done);
  return py::make_tuple(ids, dists);
}

void check_k(std::int64_t k) {
  if (k < 1) throw py::value_error("k must be at least 1");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "vecsearch: HNSW approximate nearest neighbor search (C++20, SIMD)";
  m.attr("__version__") = version();
  m.def(
      "active_kernels", [] { return std::string(active_kernels().name); },
      "Name of the distance kernels in use: 'avx2', 'neon' or 'scalar'.");

  py::class_<FlatIndex>(m, "FlatIndex", "Exact brute-force index. Ids are insertion order.")
      .def(py::init([](std::size_t dim, const std::string& metric) {
             return FlatIndex(dim, parse_metric(metric));
           }),
           py::arg("dim"), py::arg("metric") = "l2")
      .def(
          "add",
          [](FlatIndex& self, const py::handle& vectors) {
            const Matrix x = as_matrix(vectors, self.dim(), "vectors");
            py::gil_scoped_release release;
            self.add(x.data, x.rows);
          },
          py::arg("vectors"), "Append vectors, shape (n, dim) float32. Ids continue from len().")
      .def(
          "search",
          [](const FlatIndex& self, const py::handle& queries, std::int64_t k,
             std::size_t num_threads) {
            check_k(k);
            const Matrix q = as_matrix(queries, self.dim(), "queries");
            SearchResult r;
            {
              py::gil_scoped_release release;
              r = self.search(q.data, q.rows, static_cast<std::size_t>(k), num_threads);
            }
            return to_numpy(std::move(r));
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("num_threads") = 0,
          "Returns (ids int64 (nq, k), distances float32 (nq, k)); missing results are -1 / inf.")
      .def("delete", &FlatIndex::remove, py::arg("id"), "Soft-delete an id. False if absent.")
      .def("save", &FlatIndex::save, py::arg("path"), py::call_guard<py::gil_scoped_release>())
      .def_static("load", &FlatIndex::load, py::arg("path"))
      .def("__len__", &FlatIndex::size)
      .def_property_readonly("dim", &FlatIndex::dim)
      .def_property_readonly("metric", [](const FlatIndex& s) { return metric_name(s.metric()); })
      .def_property_readonly("memory_bytes", &FlatIndex::memory_bytes);

  py::class_<HnswIndex>(m, "HNSWIndex", "HNSW approximate nearest neighbor index.")
      .def(py::init([](std::size_t dim, const std::string& metric, std::size_t M,
                       std::size_t ef_construction, std::size_t ef_search, std::uint64_t seed) {
             HnswParams p;
             p.M = M;
             p.ef_construction = ef_construction;
             p.ef_search = ef_search;
             p.seed = seed;
             return std::make_unique<HnswIndex>(dim, parse_metric(metric), p);
           }),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("ef_search") = 50, py::arg("seed") = 100)
      .def(
          "add",
          [](HnswIndex& self, const py::handle& vectors, const py::object& ids,
             std::size_t num_threads) {
            const Matrix x = as_matrix(vectors, self.dim(), "vectors");
            std::optional<Labels> labels;
            if (!ids.is_none()) {
              labels = as_labels(ids, "ids");
              if (static_cast<std::size_t>(labels->shape(0)) != x.rows)
                throw py::value_error("ids has " + std::to_string(labels->shape(0)) +
                                      " entries but vectors has " + std::to_string(x.rows) +
                                      " rows");
            }
            const std::int64_t* lp = labels ? labels->data() : nullptr;
            py::gil_scoped_release release;
            self.add(x.data, x.rows, lp, num_threads);
          },
          py::arg("vectors"), py::arg("ids") = py::none(), py::arg("num_threads") = 0,
          "Insert vectors (n, dim) float32. ids: optional int64 labels (default: insertion "
          "order). An existing id is replaced (upsert).")
      .def(
          "search",
          [](const HnswIndex& self, const py::handle& queries, std::int64_t k,
             std::optional<std::size_t> ef, std::size_t num_threads, const py::object& filter) {
            check_k(k);
            const Matrix q = as_matrix(queries, self.dim(), "queries");
            std::optional<Labels> allowed;
            if (!filter.is_none()) allowed = as_labels(filter, "filter");
            SearchResult r;
            {
              py::gil_scoped_release release;
              r = self.search(q.data, q.rows, static_cast<std::size_t>(k), ef.value_or(0),
                              num_threads, allowed ? allowed->data() : nullptr,
                              allowed ? static_cast<std::size_t>(allowed->shape(0)) : 0);
            }
            return to_numpy(std::move(r));
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = py::none(),
          py::arg("num_threads") = 0, py::arg("filter") = py::none(),
          "Returns (ids int64 (nq, k), distances float32 (nq, k)); missing results are -1 / inf. "
          "filter: optional array of allowed ids.")
      .def("delete", &HnswIndex::remove, py::arg("id"), "Soft-delete an id. False if absent.")
      .def("save", &HnswIndex::save, py::arg("path"), py::call_guard<py::gil_scoped_release>())
      .def_static("load", &HnswIndex::load, py::arg("path"))
      .def("__len__", &HnswIndex::size)
      .def("__contains__", &HnswIndex::contains)
      .def_property("ef_search", &HnswIndex::ef_search, &HnswIndex::set_ef_search)
      .def_property("prefetch", &HnswIndex::prefetch, &HnswIndex::set_prefetch)
      .def_property_readonly("dim", &HnswIndex::dim)
      .def_property_readonly("metric", [](const HnswIndex& s) { return metric_name(s.metric()); })
      .def_property_readonly("M", [](const HnswIndex& s) { return s.params().M; })
      .def_property_readonly("ef_construction",
                             [](const HnswIndex& s) { return s.params().ef_construction; })
      .def_property_readonly("element_count", &HnswIndex::element_count)
      .def_property_readonly("memory_bytes", &HnswIndex::memory_bytes);
}
