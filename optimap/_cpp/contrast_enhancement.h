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


template <typename Derived, typename Derived2>
auto minmax_masked(const Derived &block, const Derived2 &mask) {
  assert(mask.shape() == block.shape());

  using Scalar = typename Derived::value_type;
  Scalar max   = std::numeric_limits<Scalar>::lowest();
  Scalar min   = std::numeric_limits<Scalar>::max();

  for (int row = 0; row < mask.shape(0); row++) {
    for (int col = 0; col < mask.shape(1); col++) {
      if (mask(row, col)) {
        max = std::max(max, block(row, col));
        min = std::min(min, block(row, col));
      }
    }
  }
  return std::make_pair(min, max);
}

auto genCircleMask(const int kernel_size) {
  const bool is_odd = static_cast<bool>(kernel_size % 2);
  if (!is_odd) {
    return Mask2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, true);
  }

  const Vec2i center         = {kernel_size / 2, kernel_size / 2};
  const unsigned int radius2 = (kernel_size / 2 + 1) * (kernel_size / 2);

  Mask2b circleblock = Mask2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, false);
  for (long row = 0; row < kernel_size; row++) {
    for (long col = 0; col < kernel_size; col++) {
      const auto v          = Vec2i(row, col) - center;
      circleblock(row, col) = v.length_squared() <= radius2;
    }
  }

  return circleblock;
}

auto genCircleMask2(const int kernel_size) {
  const bool is_odd = static_cast<bool>(kernel_size % 2);
  if (!is_odd) {

    return Mask2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, true);
  }

  const Vec2i center         = {kernel_size / 2, kernel_size / 2};
  const unsigned int radius2 = (kernel_size / 2) * (kernel_size / 2);

  Mask2b circleblock = Mask2b({static_cast<std::size_t>(kernel_size), static_cast<std::size_t>(kernel_size)}, false);
  for (long row = 0; row < kernel_size; row++) {
    for (long col = 0; col < kernel_size; col++) {
      const auto v          = Vec2i(row, col) - center;
      circleblock(row, col) = v.length_squared() <= radius2;
    }
  }

  return circleblock;
}

auto circleMaskFactory(std::size_t kernel_size, std::string kernel_type) {
  Mask2b mask;
  if (kernel_type == "block") {
    mask = Mask2b({kernel_size, kernel_size}, true);
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
void _contrast_enhancement_padded(const Array2f &img, const Mask2b &mask, T &out) {
  assert(img.dimension() == out.dimension());
  assert(img.shape(0) == out.shape(0) && img.shape(1) == out.shape(1));
  assert(mask.shape(0) == mask.shape(1));

  const int Nx          = img.shape(0);
  const int Ny          = img.shape(1);
  const int kernel_size = mask.shape(0);

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
      const auto maskv = xt::view(mask, xt::range(mstartx, mendx), xt::range(mstarty, mendy));

      const auto [min, max] = minmax_masked(block, maskv);
      if (max != min) {
        out(row, col) = (img(row, col) - min) / (max - min);
      } else {
        out(row, col) = 0;
      }
    }
  }
}

Array2f contrast_enhancement_img(const Array2f &img,
                                 std::size_t kernel_size,
                                 std::string kernel_type) {
  if (bool is_odd = kernel_size % 2; !is_odd) {
    throw std::runtime_error("only odd kernel sizes supported!");
  }

  Array2f out(img.shape(), std::numeric_limits<float>::quiet_NaN());
  auto mask = circleMaskFactory(kernel_size, kernel_type);
  _contrast_enhancement_padded(img, mask, out);

  return out;
}

Array3f contrast_enhancement_video(const Array3f &video,
                                   std::size_t kernel_size,
                                   std::string kernel_type) {
  if (bool is_odd = kernel_size % 2; !is_odd) {
    throw std::runtime_error("only odd kernel sizes supported!");
  }

  Array3f out(video.shape(), std::numeric_limits<float>::quiet_NaN());
  const auto mask = circleMaskFactory(kernel_size, kernel_type);
  const auto Nt   = video.shape(0);
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int t = 0; t < Nt; t++) {
    auto img     = xt::view(video, t, xt::all(), xt::all());
    auto out_img = xt::view(out, t, xt::all(), xt::all());
    _contrast_enhancement_padded(img, mask, out_img);
  }

  return out;
}