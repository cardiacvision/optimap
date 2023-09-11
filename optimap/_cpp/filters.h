#pragma once

#ifdef USE_OMP
#include <omp.h>
#endif

#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "vec2.h"

#include "definitions.h"

Array3f normalizePixelwiseSlidingWindow(const Array3f &video, const int wt, const float ymin = 0, const float ymax = 1) {
  const auto Nt = video.shape(0);
  const auto Nx = video.shape(1);
  const auto Ny = video.shape(2);

  Array3f result(video.shape(), NAN);
  const auto eps = std::numeric_limits<float>::epsilon();

#pragma omp parallel for collapse(2)
  for (int kx = 0; kx < Nx; kx++) {
    for (int ky = 0; ky < Ny; ky++) {
      for (int kt = 0; kt < Nt; kt++) {
        float maxp = std::numeric_limits<float>::lowest();
        float minp = std::numeric_limits<float>::max();

        for (int kti = -wt; kti <= wt; kti++) {
          // prevent index overrun
          if ((kt + kti) >= 0 && (kt + kti) < Nt) {
            auto val = video(kt + kti, kx, ky);
            if (val > maxp) maxp = val;
            if (val < minp) minp = val;
          }
        }

        result(kt, kx, ky) = (video(kt, kx, ky) - minp) / (maxp - minp + eps) * (ymax - ymin) + ymin;
      }
    }
  }

  return result;
}

Array2f createmoothingFilterKernel(size_t ksize, double sigma, Vec2 offset) {
  // set standard deviation to 1.0

  double r, s = 2.0 * sigma * sigma;

  Array2f gKernel = Array2f({ksize, ksize});
  if ((gKernel.shape(0) != ksize) && (gKernel.shape(1) != ksize)) {
    throw std::runtime_error("ksize");
  }

  // sum is for normalization
  double sum = 0.0;

  int krad = ((int)ksize - 1) / 2;

  double offx = offset.f[0];
  double offy = offset.f[1];

  // generate 5x5 kernel
  for (int x = -krad; x <= krad; x++) {
    for (int y = -krad; y <= krad; y++) {
      r                           = std::sqrt((x - offx) * (x - offx) + (y - offy) * (y - offy));
      gKernel(x + krad, y + krad) = (std::exp(-(r * r) / s)) / (M_PI * s);
      sum += gKernel(x + krad, y + krad);
    }
  }

  // normalize the Kernel
  gKernel /= sum;

  return gKernel;
}

//template <size_t ksize>
Array3f spatiotemporalSmoothingFilter(
    const Array3f &video, std::size_t ksize, int wx, int wy, int wt) {
  // treats boundaries differently, has a gaussian kernel

  const auto Nt = video.shape(0);
  const auto Nx = video.shape(1);
  const auto Ny = video.shape(2);

  Array3f result(video.shape(), NAN);

  const Vec2 offset  = Vec2(0.0, 0.0);
  const float gsigma = (float)wx / 2.f;
  auto kernel        = createmoothingFilterKernel(ksize, gsigma, offset);

  //check whether cumulative sum of deviations is larger than threshold
#pragma omp parallel for collapse(2)
  for (int kt = 0; kt < Nt; kt++) {
    for (int ky = 0; ky < Ny; ky++) {
      for (int kx = 0; kx < Nx; kx++) {
        float pixel_xyt      = video(kt, kx, ky);
        float pixel_xyt_filt = std::isnan(pixel_xyt) ? 0 : pixel_xyt;

        for (int kti = -wt; kti <= wt; kti++) {
          for (int kxi = -wx; kxi <= wx; kxi++) {
            for (int kyi = -wy; kyi <= wy; kyi++) {

              if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kt + kti) >= 0 && (kx + kxi) < Nx &&
                  (ky + kyi) < Ny && (kt + kti) < Nt) {
                if (std::isnan(video(kt + kti, kx + kxi, ky + kyi))) {
                } else {

                  float d = std::sqrt(static_cast<float>(kxi * kxi + kyi * kyi));
                  if (d <= static_cast<float>(wx)) {
                    if (kxi == 0 && kyi == 0 && kti == 0) {
                    } else {
                      pixel_xyt_filt +=
                          video(kt + kti, kx + kxi, ky + kyi) * kernel(kxi + wx, kyi + wy);
                    }
                  }
                }
              }
            }
          }
        }
        result(kt, kx, ky) = pixel_xyt_filt * 1.f / static_cast<float>(wt * 2 + 1);
      }
    }
  }  // t-loop

  return result;
}