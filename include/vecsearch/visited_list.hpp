#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecsearch {

// "Have I seen node i during this query?" in O(1), with O(1) reset between queries.
//
// Instead of clearing an n-element bitmap before every query (O(n), which would dominate for
// large indexes), each query gets a new epoch number and node i counts as visited iff
// marks[i] == epoch. Only when the 16-bit epoch wraps around (every 65535 queries) is the array
// actually cleared.
class VisitedList {
 public:
  // Starts a new query over node ids [0, n).
  void reset(std::size_t n) {
    if (marks_.size() < n) marks_.resize(n, 0);  // new slots are 0, never equal to the epoch
    if (++epoch_ == 0) {
      std::fill(marks_.begin(), marks_.end(), 0);
      epoch_ = 1;
    }
  }

  // Marks id as visited; returns true if it already was.
  bool test_and_set(std::uint32_t id) {
    if (marks_[id] == epoch_) return true;
    marks_[id] = epoch_;
    return false;
  }

  bool contains(std::uint32_t id) const { return marks_[id] == epoch_; }
  const std::uint16_t* address(std::uint32_t id) const { return marks_.data() + id; }

 private:
  std::vector<std::uint16_t> marks_;
  std::uint16_t epoch_ = 0;
};

}  // namespace vecsearch
