#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecsearch {

// A (distance, id) pair. Ordered by distance, ties broken by id so results are deterministic.
template <class Id>
struct Scored {
  float dist;
  Id id;
  friend bool operator<(const Scored& a, const Scored& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
  }
  friend bool operator>(const Scored& a, const Scored& b) { return b < a; }
};

// Keeps the k smallest items seen so far in a bounded max-heap.
//
// The root is the current k-th best (the worst kept item), so deciding whether a new candidate
// belongs costs one comparison, and inserting costs O(log k). For n candidates: O(n log k) time,
// O(k) memory, versus O(n log n) and O(n) for sorting everything.
template <class Id>
class TopK {
 public:
  explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

  void push(float dist, Id id) {
    if (k_ == 0) return;
    const Scored<Id> item{dist, id};
    if (heap_.size() < k_) {
      heap_.push_back(item);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (item < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = item;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }

  bool full() const { return heap_.size() == k_; }
  std::size_t size() const { return heap_.size(); }
  // Distance of the worst kept item (only meaningful when size() > 0).
  float worst() const { return heap_.front().dist; }

  // Returns the kept items sorted best-first and empties the heap.
  std::vector<Scored<Id>> take_sorted() {
    std::sort_heap(heap_.begin(), heap_.end());
    return std::move(heap_);
  }

 private:
  std::size_t k_;
  std::vector<Scored<Id>> heap_;
};

}  // namespace vecsearch
