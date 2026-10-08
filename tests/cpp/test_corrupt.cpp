// Corrupt and truncated index files must make load() throw std::runtime_error. They must never
// load "successfully" and then crash in search. Run under the sanitizer preset too.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/hnsw_index.hpp"

using namespace vecsearch;

namespace {

using Bytes = std::vector<char>;
using U64 = std::uint64_t;

// Byte offsets of the HNSW file header (see HnswIndex::save).
constexpr std::size_t kDimOff = 12;
constexpr std::size_t kMOff = 21;
constexpr std::size_t kCountOff = 54;
constexpr std::size_t kMaxLevelOff = 62;
constexpr std::size_t kEntryOff = 66;
constexpr std::size_t kLabelsOff = 70;

std::string temp_path(const std::string& name) {
  return vecsearch::test::unique_temp_path(name);
}

Bytes read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return Bytes(std::istreambuf_iterator<char>(in), {});
}

void write_file(const std::string& path, const Bytes& b) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

template <class T>
void patch(Bytes& b, std::size_t off, T value) {
  ASSERT_LE(off + sizeof(T), b.size());
  std::memcpy(b.data() + off, &value, sizeof(T));
}

template <class T>
T peek(const Bytes& b, std::size_t off) {
  T v;
  std::memcpy(&v, b.data() + off, sizeof(T));
  return v;
}

Bytes saved_hnsw(const HnswIndex& index) {
  const auto path = temp_path("vecsearch_corrupt_src.bin");
  index.save(path);
  Bytes b = read_file(path);
  std::filesystem::remove(path);
  return b;
}

// Loads the bytes and, if that succeeds, searches: either step may throw, neither may crash.
void load_and_search(const Bytes& b, std::size_t dim) {
  const auto path = temp_path("vecsearch_corrupt.bin");
  write_file(path, b);
  struct Cleanup {
    std::string p;
    ~Cleanup() { std::filesystem::remove(p); }
  } cleanup{path};
  const auto index = HnswIndex::load(path);
  const auto q = test::random_vectors(3, dim, 99);
  index->search(q.data(), 3, 5, 20, 1);
}

struct Layout {
  std::size_t n, dim, M, M0;
  std::size_t levels_off() const { return kLabelsOff + 8 * n; }
  std::size_t level0_off() const { return kLabelsOff + 10 * n; }
  std::size_t block0(std::uint32_t id) const { return level0_off() + id * (1 + M0) * 4; }
};

class CorruptHnsw : public ::testing::Test {
 protected:
  static constexpr std::size_t kN = 60, kDim = 4, kM = 4;
  void SetUp() override {
    data = test::random_vectors(kN, kDim, 5);
    index = std::make_unique<HnswIndex>(kDim, Metric::L2, HnswParams{.M = kM});
    index->add(data.data(), kN, nullptr, 1);
    bytes = saved_hnsw(*index);
    lay = {kN, kDim, kM, 2 * kM};
  }
  std::vector<float> data;
  std::unique_ptr<HnswIndex> index;
  Bytes bytes;
  Layout lay{};
};

}  // namespace

TEST_F(CorruptHnsw, UnmodifiedFileLoads) {
  EXPECT_NO_THROW(load_and_search(bytes, kDim));
}

// The bug report: a one-node index whose max level is changed from 0 to 1 used to load and then
// crash in search (layer 1 of a node that has no layer-1 storage).
TEST(CorruptHnswSmall, OneNodeMaxLevelRaised) {
  HnswIndex index(4, Metric::L2);
  const float v[4] = {1, 2, 3, 4};
  index.add(v, 1);
  const int level = index.level_of(0);
  Bytes b = saved_hnsw(index);
  ASSERT_EQ(peek<std::int32_t>(b, kMaxLevelOff), level);
  patch<std::int32_t>(b, kMaxLevelOff, level + 1);  // a layer the node has no storage for
  EXPECT_THROW(load_and_search(b, 4), std::runtime_error);
}

TEST(CorruptHnswSmall, EmptyIndexMustBeEmptyState) {
  HnswIndex index(4, Metric::L2);
  Bytes b = saved_hnsw(index);
  EXPECT_NO_THROW(load_and_search(b, 4));
  Bytes bad = b;
  patch<std::int32_t>(bad, kMaxLevelOff, 0);  // claims an entry point that does not exist
  EXPECT_THROW(load_and_search(bad, 4), std::runtime_error);
  bad = b;
  patch<std::uint32_t>(bad, kEntryOff, 3);
  EXPECT_THROW(load_and_search(bad, 4), std::runtime_error);
}

TEST_F(CorruptHnsw, MaxLevelOutOfRange) {
  for (std::int32_t level : {-1, -7, 32, 1000}) {
    Bytes b = bytes;
    patch<std::int32_t>(b, kMaxLevelOff, level);
    EXPECT_THROW(load_and_search(b, kDim), std::runtime_error) << level;
  }
  Bytes b = bytes;  // in range but not the highest node level
  patch<std::int32_t>(b, kMaxLevelOff, index->max_level() + 1);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, EntryPointInvalid) {
  Bytes b = bytes;
  patch<std::uint32_t>(b, kEntryOff, kN);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);

  ASSERT_GT(index->max_level(), 0);
  std::uint32_t low = 0;
  while (index->level_of(low) == index->max_level()) ++low;
  b = bytes;  // exists, but is not on the top layer
  patch<std::uint32_t>(b, kEntryOff, low);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, LevelAboveMax) {
  Bytes b = bytes;
  b[lay.levels_off() + 3] = static_cast<char>(40);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, EdgeTargetOutOfRange) {
  Bytes b = bytes;
  ASSERT_GE(peek<std::uint32_t>(b, lay.block0(0)), 1u);
  patch<std::uint32_t>(b, lay.block0(0) + 4, kN);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, UpperEdgeToNodeNotOnThatLayer) {
  // Find a layer-1 node with at least one layer-1 link and a node that is only on layer 0.
  std::size_t upper_off = lay.level0_off() + kN * (1 + lay.M0) * 4;
  std::size_t block = 0;
  for (std::uint32_t id = 0; id < kN; ++id) {
    if (index->level_of(id) >= 1 && !index->neighbors(id, 1).empty()) {
      block = upper_off;
      break;
    }
    upper_off += index->level_of(id) * (1 + kM) * 4;
  }
  ASSERT_NE(block, 0u);
  std::uint32_t ground = 0;
  while (index->level_of(ground) != 0) ++ground;
  Bytes b = bytes;
  patch<std::uint32_t>(b, block + 4, ground);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, NeighborCountAboveCapacity) {
  Bytes b = bytes;
  patch<std::uint32_t>(b, lay.block0(5), static_cast<std::uint32_t>(lay.M0 + 1));
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

TEST_F(CorruptHnsw, LabelsReservedOrDuplicated) {
  Bytes b = bytes;
  patch<std::int64_t>(b, kLabelsOff + 8 * 2, -1);
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
  b = bytes;
  patch<std::int64_t>(b, kLabelsOff + 8 * 2, peek<std::int64_t>(b, kLabelsOff + 8 * 3));
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

// Header sizes that would overflow or demand far more memory than the file holds must be
// rejected before anything is allocated.
TEST_F(CorruptHnsw, HeaderSizes) {
  const U64 counts[] = {U64{kN + 1}, U64{1} << 32, U64{1} << 40, ~U64{0}};
  for (U64 count : counts) {
    Bytes b = bytes;
    patch<std::uint64_t>(b, kCountOff, count);
    EXPECT_THROW(load_and_search(b, kDim), std::runtime_error) << count;
  }
  const U64 dims[] = {U64{0}, U64{17}, U64{1} << 40, ~U64{0}};
  for (U64 dim : dims) {
    Bytes b = bytes;
    patch<std::uint64_t>(b, kDimOff, dim);
    EXPECT_THROW(load_and_search(b, kDim), std::runtime_error) << dim;
  }
  const U64 ms[] = {U64{0}, U64{1}, U64{5}, U64{1} << 62, ~U64{0}};
  for (U64 m : ms) {
    Bytes b = bytes;
    patch<std::uint64_t>(b, kMOff, m);
    EXPECT_THROW(load_and_search(b, kDim), std::runtime_error) << m;
  }
}

TEST_F(CorruptHnsw, EveryTruncationThrows) {
  for (std::size_t len = 0; len < bytes.size(); ++len) {
    const Bytes b(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(len));
    EXPECT_THROW(load_and_search(b, kDim), std::runtime_error) << "length " << len;
  }
}

TEST_F(CorruptHnsw, TrailingBytesThrow) {
  Bytes b = bytes;
  b.push_back('x');
  EXPECT_THROW(load_and_search(b, kDim), std::runtime_error);
}

// Random single-byte corruption anywhere: load may succeed (e.g. a changed vector value) or
// throw, but must never crash or read out of bounds (ASan checks the latter).
TEST_F(CorruptHnsw, RandomByteFlipsNeverCrash) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 3000; ++trial) {
    Bytes b = bytes;
    const std::size_t pos = rng() % b.size();
    b[pos] = static_cast<char>(b[pos] ^ static_cast<char>(1 + rng() % 255));
    try {
      load_and_search(b, kDim);
    } catch (const std::runtime_error&) {
    }
  }
}

// ---- FlatIndex ----

TEST(CorruptFlat, TruncationTrailingAndSizes) {
  FlatIndex flat(5, Metric::L2);
  const auto data = test::random_vectors(7, 5, 3);
  flat.add(data.data(), 7);
  const auto path = temp_path("vecsearch_corrupt_flat.bin");
  flat.save(path);
  const Bytes bytes = read_file(path);

  auto try_load = [&](const Bytes& b) {
    write_file(path, b);
    const FlatIndex f = FlatIndex::load(path);
    f.search(data.data(), 1, 3, 1);
  };
  EXPECT_NO_THROW(try_load(bytes));
  for (std::size_t len = 0; len < bytes.size(); ++len) {
    EXPECT_THROW(try_load(Bytes(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(len))),
                 std::runtime_error)
        << "length " << len;
  }
  Bytes b = bytes;
  b.push_back(0);
  EXPECT_THROW(try_load(b), std::runtime_error);

  // Flat header: magic(8) version(4) dim u64 @12, metric u8 @20, size u64 @21.
  const U64 values[] = {U64{0}, U64{1} << 40, ~U64{0}};
  for (U64 v : values) {
    b = bytes;
    patch<std::uint64_t>(b, 12, v);
    EXPECT_THROW(try_load(b), std::runtime_error) << "dim " << v;
    b = bytes;
    patch<std::uint64_t>(b, 21, v == 0 ? 8 : v);
    EXPECT_THROW(try_load(b), std::runtime_error) << "size " << v;
  }
  b = bytes;
  b[b.size() - 1] = 7;  // deleted flag must be 0 or 1
  EXPECT_THROW(try_load(b), std::runtime_error);
  std::filesystem::remove(path);
}
