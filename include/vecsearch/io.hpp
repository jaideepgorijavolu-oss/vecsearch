#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace vecsearch::io {

// Small helpers for the binary index formats. Files are written in host byte order; the
// version header lets a future reader reject (or convert) files it does not understand.

template <class T>
void write_pod(std::ofstream& out, const T& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T read_pod(std::ifstream& in) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("index file is truncated");
  return v;
}

template <class T>
void write_array(std::ofstream& out, const T* data, std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

template <class T>
void read_array(std::ifstream& in, T* data, std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error("index file is truncated");
}

inline std::ofstream open_out(const std::string& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open '" + path + "' for writing");
  return out;
}

inline std::ifstream open_in(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "' for reading");
  return in;
}

// Thrown for any malformed file. Python sees it as RuntimeError.
[[noreturn]] inline void corrupt(const std::string& why) {
  throw std::runtime_error("corrupt index file: " + why);
}

// Bytes left between the read position and the end of the file.
inline std::uint64_t remaining_bytes(std::ifstream& in) {
  const auto pos = in.tellg();
  in.seekg(0, std::ios::end);
  const auto end = in.tellg();
  in.seekg(pos);
  if (!in || pos < 0 || end < pos) corrupt("cannot determine file size");
  return static_cast<std::uint64_t>(end - pos);
}

// Size arithmetic on untrusted header values: throw instead of wrapping around.
inline std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
  std::uint64_t r;
  if (__builtin_mul_overflow(a, b, &r)) corrupt("size field overflows");
  return r;
}
inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t r;
  if (__builtin_add_overflow(a, b, &r)) corrupt("size field overflows");
  return r;
}

// Upper bounds for header fields, far above anything real, so later arithmetic cannot overflow.
inline constexpr std::uint64_t kMaxFileDim = std::uint64_t{1} << 20;
inline constexpr std::uint64_t kMaxFileM = std::uint64_t{1} << 16;

// An 8-character magic string followed by a uint32 format version.
inline void write_header(std::ofstream& out, const char (&magic)[9], std::uint32_t version) {
  out.write(magic, 8);
  write_pod(out, version);
}

inline void check_header(std::ifstream& in, const char (&magic)[9], std::uint32_t version) {
  char got[8];
  in.read(got, 8);
  if (!in || std::string(got, 8) != std::string(magic, 8))
    throw std::runtime_error("not a vecsearch index file of this type (bad magic)");
  const auto v = read_pod<std::uint32_t>(in);
  if (v != version)
    throw std::runtime_error("unsupported index file version " + std::to_string(v) + " (expected " +
                             std::to_string(version) + ")");
}

}  // namespace vecsearch::io
