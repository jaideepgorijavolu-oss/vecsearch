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
