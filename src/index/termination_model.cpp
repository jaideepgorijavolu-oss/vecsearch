#include "vecsearch/termination_model.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace vecsearch {

namespace {

constexpr char kHeader[] = "vecsearch-termination-model";
constexpr int kFormatVersion = 1;
constexpr std::size_t kMaxQueryDim = 4096;
constexpr std::size_t kMaxNodes = 1 << 22;

[[noreturn]] void bad(const std::string& what) {
  throw std::runtime_error("invalid termination model: " + what);
}

template <class T>
T read(std::istream& in, const char* what) {
  T v;
  if (!(in >> v)) bad(std::string("cannot read ") + what);
  return v;
}

void expect(std::istream& in, const std::string& word) {
  if (read<std::string>(in, word.c_str()) != word) bad("expected '" + word + "'");
}

}  // namespace

TerminationModel TerminationModel::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  return parse(in);
}

// Format (whitespace separated):
//   vecsearch-termination-model 1
//   kind linear|gbdt   checkpoint <n>   query_dim <d>   bias <b>
//   linear: weights <n> w_0 .. w_{n-1}
//   gbdt:   trees <t>, then per tree: nodes <m>, then m x (feature threshold left right value)
//   end
TerminationModel TerminationModel::parse(std::istream& in) {
  TerminationModel m;
  expect(in, kHeader);
  if (read<int>(in, "version") != kFormatVersion) bad("unsupported version");
  expect(in, "kind");
  const auto kind = read<std::string>(in, "kind");
  if (kind != "linear" && kind != "gbdt") bad("unknown kind '" + kind + "'");
  m.linear_ = kind == "linear";
  expect(in, "checkpoint");
  m.checkpoint_ = read<std::size_t>(in, "checkpoint");
  if (m.checkpoint_ == 0) bad("checkpoint must be positive");
  expect(in, "query_dim");
  m.query_dim_ = read<std::size_t>(in, "query_dim");
  if (m.query_dim_ > kMaxQueryDim) bad("query_dim too large");
  expect(in, "bias");
  m.bias_ = read<double>(in, "bias");
  if (!std::isfinite(m.bias_)) bad("non-finite bias");
  const std::size_t num_inputs = kNumTerminationFeatures + m.query_dim_;

  if (m.linear_) {
    expect(in, "weights");
    if (read<std::size_t>(in, "weight count") != num_inputs) bad("weight count mismatch");
    m.weights_.resize(num_inputs);
    for (auto& w : m.weights_) {
      w = read<double>(in, "weight");
      if (!std::isfinite(w)) bad("non-finite weight");
    }
  } else {
    expect(in, "trees");
    const auto trees = read<std::size_t>(in, "tree count");
    if (trees == 0 || trees > kMaxNodes) bad("tree count out of range");
    for (std::size_t t = 0; t < trees; ++t) {
      expect(in, "nodes");
      const auto n = read<std::size_t>(in, "node count");
      if (n == 0 || m.nodes_.size() + n > kMaxNodes) bad("node count out of range");
      const auto root = static_cast<std::int32_t>(m.nodes_.size());
      m.tree_roots_.push_back(static_cast<std::uint32_t>(root));
      for (std::size_t i = 0; i < n; ++i) {
        Node node{};
        node.feature = read<std::int32_t>(in, "feature");
        node.threshold = read<double>(in, "threshold");
        node.left = read<std::int32_t>(in, "left");
        node.right = read<std::int32_t>(in, "right");
        node.value = read<double>(in, "value");
        if (node.feature >= 0) {
          // Children must come after their parent within the same tree: no cycles, no
          // out-of-range jumps, so prediction always terminates.
          const auto self = static_cast<std::int32_t>(i);
          if (static_cast<std::size_t>(node.feature) >= num_inputs) bad("feature out of range");
          if (node.left <= self || node.right <= self || static_cast<std::size_t>(node.left) >= n ||
              static_cast<std::size_t>(node.right) >= n)
            bad("child index out of range");
          if (std::isnan(node.threshold)) bad("NaN threshold");
          node.left += root;
          node.right += root;
        } else {
          if (node.feature != -1) bad("bad leaf marker");
          if (!std::isfinite(node.value)) bad("non-finite leaf value");
        }
        m.nodes_.push_back(node);
      }
    }
  }
  expect(in, "end");
  return m;
}

double TerminationModel::predict_log_evals(const float* features, const float* query) const {
  double y = bias_;
  if (linear_) {
    for (std::size_t i = 0; i < weights_.size(); ++i) y += weights_[i] * input(i, features, query);
    return y;
  }
  for (const std::uint32_t root : tree_roots_) {
    const Node* node = &nodes_[root];
    while (node->feature >= 0) {
      const double x = input(static_cast<std::size_t>(node->feature), features, query);
      node = &nodes_[x <= node->threshold ? node->left : node->right];
    }
    y += node->value;
  }
  return y;
}

}  // namespace vecsearch
