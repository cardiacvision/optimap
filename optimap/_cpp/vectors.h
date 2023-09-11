#pragma once

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>

/* Forward Definitions */
template <typename T, size_t N>
class VectorN;
template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Vector4;

/* Aliases */
using Vec2i = Vector2<int>;
using Vec2f = Vector2<float>;
using Vec2d = Vector2<double>;

using Vec3f = Vector3<float>;
using Vec4f = Vector4<float>;

/* Implementations */

// A minimal N-dimensional vector class of floats/integers
template <typename T, size_t N>
class VectorN {
 public:
  std::array<T, N> f = {};

  constexpr VectorN() = default;

  template <typename D>
  explicit VectorN(const VectorN<D, N> &v) {
    for (int i = 0; i < N; i++) {
      this->f[i] = static_cast<T>(v.f[i]);
    }
  }

  T &operator[](const int i) { return f[i]; }

  T operator[](const int i) const { return f[i]; }

  bool operator==(const VectorN &v) const { return (v.f == f); }

  bool operator!=(const VectorN &v) const { return (v.f != f); }

  void operator+=(const VectorN &v) {
    for (int i = 0; i < N; i++) {
      this->f[i] += v[i];
    }
  }

  void operator-=(const VectorN &v) {
    for (int i = 0; i < N; i++) {
      this->f[i] -= v[i];
    }
  }

  void operator*=(const T &a) {
    for (int i = 0; i < N; i++) {
      this->f[i] *= a;
    }
  }

  void operator/=(const T &a) {
    for (int i = 0; i < N; i++) {
      this->f[i] /= a;
    }
  }

  const T *data() const { return this->f.data(); }
  operator const T *() const { return data(); }

  T *data() { return this->f.data(); }
  operator T *() { return data(); }

  T length_squared() const { return inner_product(*this); }

  [[nodiscard]] float length() const { return std::sqrt(this->length_squared()); }

  T element_product() const {
    return std::accumulate(f.begin(), f.end(), T{}, std::multiplies<T>());
  }

  template <typename D>
  T inner_product(const VectorN<D, N> &o) const {
    return std::inner_product(f.begin(), f.end(), o.f.begin(), static_cast<T>(0));
  }

  // basic iterators
  using value_type = T;
  using iterator   = typename std::array<T, N>::iterator;
  iterator begin() { return f.begin(); }
  iterator end() { return f.end(); }

  using const_iterator = typename std::array<T, N>::const_iterator;
  const_iterator begin() const { return f.begin(); }
  const_iterator end() const { return f.end(); }
};

template <typename T>
class Vector2 : public VectorN<T, 2> {
 public:
  constexpr Vector2() = default;

  constexpr Vector2(const T &x, const T &y) { this->f = {{x, y}}; }

  using VectorN<T, 2>::VectorN;

  Vector2 operator/(const T &a) const { return Vector2(this->f[0] / a, this->f[1] / a); }

  Vector2 operator-(const Vector2 &v) const { return Vector2(this->f[0] - v[0], this->f[1] - v[1]); }

  Vector2 operator+(const Vector2 &v) const { return Vector2(this->f[0] + v[0], this->f[1] + v[1]); }

  template <typename D>
  Vector2 operator*(const D &a) const {
    return Vector2(this->f[0] * a, this->f[1] * a);
  }

  Vector2 operator-() const { return Vector2(-this->f[0], -this->f[1]); }
};


template <typename T>
class Vector3 : public VectorN<T, 3> {
 public:
  constexpr Vector3() = default;

  constexpr Vector3(const T &x, const T &y, const T &z) { this->f = {{x, y, z}}; }

  using VectorN<T, 3>::VectorN;

  Vector3 operator/(const T &a) const {
    return Vector3(this->f[0] / a, this->f[1] / a, this->f[2] / a);
  }

  Vector3 operator-(const Vector3 &v) const {
    return Vector3(this->f[0] - v[0], this->f[1] - v[1], this->f[2] - v[2]);
  }

  Vector3 operator+(const Vector3 &v) const {
    return Vector3(this->f[0] + v[0], this->f[1] + v[1], this->f[2] + v[2]);
  }

  template <typename D>
  Vector3 operator*(const D &a) const {
    return Vector3(this->f[0] * a, this->f[1] * a, this->f[2] * a);
  }

  Vector3 operator-() const { return Vector3(-this->f[0], -this->f[1], -this->f[2]); }

  Vector3 cross(const Vector3 &v) const {
    return Vector3(this->f[1] * v[2] - this->f[2] * v[1], this->f[2] * v[0] - this->f[0] * v[2],
                   this->f[0] * v[1] - this->f[1] * v[0]);
  }

  T dot(const Vector3 &v) const { return this->f[0] * v[0] + this->f[1] * v[1] + this->f[2] * v[2]; }

  // Rodrigues' rotation formula
  // k: a unit vector describing the axis of rotation
  // theta: the angle (in radians) that v rotates around k
  Vector3 rotate(const Vector3 &k, double theta) {

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    return (*this * cos_theta) + k.cross(*this) * sin_theta + k * (k.dot(*this) * (1 - cos_theta));
  }

  Vector3 cwiseProduct(const Vector3 &v) const {
    return Vector3(this->f[0] * v[0], this->f[1] * v[1], this->f[2] * v[2]);
  }
};

template <typename T>
class Vector4 : public VectorN<T, 4> {
 public:
  using VectorN<T, 4>::VectorN;

  constexpr Vector4() = default;

  constexpr Vector4(const T &x, const T &y, const T &z, const T &w) { this->f = {{x, y, z, w}}; }

  Vector4 operator/(const T &a) const {
    return Vector4(this->f[0] / a, this->f[1] / a, this->f[2] / a, this->f[3] / a);
  }

  Vector4 operator-(const Vector4 &v) const {
    return Vector4(this->f[0] - v[0], this->f[1] - v[1], this->f[2] - v[2], this->f[3] - v[3]);
  }

  Vector4 operator+(const Vector4 &v) const {
    return Vector4(this->f[0] + v[0], this->f[1] + v[1], this->f[2] + v[2], this->f[3] + v[3]);
  }

  template <typename D>
  Vector4 operator*(const D &a) const {
    return Vector4(this->f[0] * a, this->f[1] * a, this->f[2] * a, this->f[3] * a);
  }

  Vector4 operator-() const { return Vector4(-this->f[0], -this->f[1], -this->f[2], -this->f[3]); }
};

template <typename T>
Vector3<T> operator*(const T &a, const Vector3<T> &v) {
  return v * a;
}

namespace detail {
  // see https://stackoverflow.com/questions/2590677
  template <class T>
  inline void hash_combine(std::size_t &seed, const T &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
}  // namespace detail

// hash functions for std::unordered_set etc.
namespace std {
  template <typename T, size_t N>
  struct hash<VectorN<T, N>> {
    inline std::size_t operator()(const VectorN<T, N> &v) const {
      std::size_t seed = 0;

      for (int i = 0; i < N; i++) {
        // see https://stackoverflow.com/questions/19966041 for the magic number
        detail::hash_combine(seed, v[i] * 2654435761);
      }
      return seed;
    }
  };

  template <typename T>
  struct hash<Vector3<T>> {
    inline std::size_t operator()(const Vector3<T> &v) const {
      std::size_t seed = 0;

      for (int i = 0; i < 3; i++) {
        detail::hash_combine(seed, v[i] * 2654435761);
      }
      return seed;
    }
  };
}  // namespace std