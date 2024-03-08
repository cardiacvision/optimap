#pragma once

#include <limits>
#include <cmath>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "definitions.h"
#include "vectors.h"

template <typename Derived, typename Derived2, typename Derived3>
auto minmax_kernel(const Derived &block, const Derived2 &kernel, const Derived3 &mask) {
  assert(kernel.shape() == block.shape());
  static_assert(std::is_same_v<typename Derived2::value_type, bool>);
  static_assert(std::is_same_v<typename Derived3::value_type, bool>);

  using Scalar = typename Derived::value_type;
  Scalar max   = std::numeric_limits<Scalar>::lowest();
  Scalar min   = std::numeric_limits<Scalar>::max();

  for (size_t row = 0; row < kernel.shape(0); row++) {
    for (size_t col = 0; col < kernel.shape(1); col++) {
      bool in_mask = mask.size() <= 1 || mask(row, col);
      if (kernel(row, col) && in_mask) {
        max = std::max(max, block(row, col));
        min = std::min(min, block(row, col));
      }
    }
  }

  if (max == std::numeric_limits<Scalar>::lowest() || min == std::numeric_limits<Scalar>::max()) {
    max = 0;
    min = 0;
  }
  return std::make_pair(min, max);
}

auto genCircleMask(const int kernel_size) {
  const bool is_odd = static_cast<bool>(kernel_size % 2);
  if (!is_odd) {
    return Array2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)},
                   true);
  }

  const Vec2i center         = {kernel_size / 2, kernel_size / 2};
  const unsigned int radius2 = (kernel_size / 2 + 1) * (kernel_size / 2);

  Array2b circleblock =
      Array2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, false);
  for (long row = 0; row < kernel_size; row++) {
    for (long col = 0; col < kernel_size; col++) {
      const auto v          = Vec2i(row, col) - center;
      circleblock(row, col) = static_cast<size_t>(v.length_squared()) <= radius2;
    }
  }

  return circleblock;
}

auto genCircleMask2(const int kernel_size) {
  const bool is_odd = static_cast<bool>(kernel_size % 2);
  if (!is_odd) {

    return Array2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)},
                   true);
  }

  const Vec2i center         = {kernel_size / 2, kernel_size / 2};
  const unsigned int radius2 = (kernel_size / 2) * (kernel_size / 2);

  Array2b circleblock =
      Array2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, false);
  for (long row = 0; row < kernel_size; row++) {
    for (long col = 0; col < kernel_size; col++) {
      const auto v          = Vec2i(row, col) - center;
      circleblock(row, col) = static_cast<size_t>(v.length_squared()) <= radius2;
    }
  }

  return circleblock;
}

auto kernel_factory(std::size_t kernel_size, std::string kernel_type) {
  Array2b mask;
  if (kernel_type == "block") {
    mask = Array2b({kernel_size, kernel_size}, true);
  } else if (kernel_type == "circle") {
    mask = genCircleMask(kernel_size);
  } else if (kernel_type == "circle2") {
    mask = genCircleMask2(kernel_size);
  } else {
    throw std::logic_error(
        "Invalid value for 'kernel_type', has to be one of ['block', 'circle', 'circle2']");
  }
  return mask;
}

template <typename T>
void _contrast_enhancement_padded(T &out,
                                  const Array2f &img,
                                  const Array2b &kernel,
                                  const Array2b &mask) {
  assert(img.dimension() == out.dimension());
  assert(img.shape(0) == out.shape(0) && img.shape(1) == out.shape(1));
  assert(kernel.shape(0) == kernel.shape(1));
  if (mask.size() > 1) {
    assert(mask.shape(0) == img.shape(0) && mask.shape(1) == img.shape(1));
  }

  const int Nx          = img.shape(0);
  const int Ny          = img.shape(1);
  const int kernel_size = kernel.shape(0);

  const int offset = (kernel_size - 1) / 2;
  for (int row = 0; row < Nx; row++) {
    for (int col = 0; col < Ny; col++) {
      int startx  = row - offset;
      int starty  = col - offset;
      int endx    = startx + kernel_size;
      int endy    = starty + kernel_size;
      int mstartx = 0, mstarty = 0, mendx = kernel_size, mendy = kernel_size;
      if (startx < 0) {
        mstartx -= startx;
        startx = 0;
      }
      if (endx > Nx) {
        mendx -= endx - Nx;
        endx = Nx;
      }
      if (starty < 0) {
        mstarty -= starty;
        starty = 0;
      }
      if (endy > Ny) {
        mendy -= endy - Ny;
        endy = Ny;
      }

      const auto block = xt::view(img, xt::range(startx, endx), xt::range(starty, endy));
      const auto kernel_block = xt::view(kernel, xt::range(mstartx, mendx), xt::range(mstarty, mendy));
      const auto mask_block = (mask.size() > 1)? xt::view(mask, xt::range(startx, endx), xt::range(starty, endy)) : Array2b();

      const bool in_mask = mask.size() <= 1 || mask(row, col);
      auto [min, max]    = minmax_kernel(block, kernel_block, mask_block);
      if (max != min && in_mask) {
        out(row, col) = (img(row, col) - min) / (max - min);
      } else {
        out(row, col) = 0;
      }
    }
  }
}

Array2f contrast_enhancement_img(const Array2f &img,
                                 std::size_t kernel_size,
                                 const Array2b &mask,
                                 std::string kernel_type) {
  if (bool is_odd = kernel_size % 2; !is_odd) {
    throw std::runtime_error("only odd kernel sizes supported!");
  }

  Array2f out(img.shape(), std::numeric_limits<float>::quiet_NaN());
  const auto kernel = kernel_factory(kernel_size, kernel_type);
  _contrast_enhancement_padded(out, img, kernel, mask);

  return out;
}

Array3f contrast_enhancement_video(const Array3f &video,
                                   std::size_t kernel_size,
                                   const Array3b &mask,
                                   std::string kernel_type) {
  if (bool is_odd = kernel_size % 2; !is_odd) {
    throw std::runtime_error("only odd kernel sizes supported!");
  }

  Array3f out(video.shape(), std::numeric_limits<float>::quiet_NaN());
  const auto kernel = kernel_factory(kernel_size, kernel_type);
  const auto Nt     = video.shape(0);
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int t = 0; t < Nt; t++) {
    auto img     = xt::view(video, t, xt::all(), xt::all());
    auto out_img = xt::view(out, t, xt::all(), xt::all());
    if (mask.size() > 1) {
      _contrast_enhancement_padded(out_img, img, kernel, xt::view(mask, t, xt::all(), xt::all()));
    } else {
      _contrast_enhancement_padded(out_img, img, kernel, Array2b());
    }
  }

  return out;
}