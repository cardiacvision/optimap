#pragma once

#include <limits>
#include <cmath>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "vec2.h"
#include "vectors.h"
#include "definitions.h"

Array4f spatiotemporalFlowFilter(
    const Array4f &displacement, int wx, int wy, int wt, const Array2f &mask) {
  // treats boundaries differently
  const auto Nt = displacement.shape(0);
  const auto Nx = displacement.shape(1);
  const auto Ny = displacement.shape(2);

  Array4f result(displacement.shape(), std::numeric_limits<float>::quiet_NaN());

  if (displacement.shape(3) != 2) {
    throw std::runtime_error("Error: array should be of type Vec2!");
  }

  const auto test_mask = [&mask](int x, int y) -> bool {
    if (mask.size() > 1) {
      return mask(x, y) > 0.5;
    } else {
      return true;
    }
  };

  const auto get_Vec2 = [&displacement](int t, int x, int y) -> Vec2 {
    return Vec2(displacement(t, x, y, 0), displacement(t, x, y, 1));
  };

  const auto set_Vec2_result = [&result](int t, int x, int y, Vec2 val) {
    result(t, x, y, 0) = val.f[0];
    result(t, x, y, 1) = val.f[1];
  };

  //check whether cumulative sum of deviations is larger than threshold
#pragma omp parallel for collapse(2)
  for (int kx = 0; kx < Nx; kx++) {
    for (int ky = 0; ky < Ny; ky++) {
      if (test_mask(kx, ky)) {
        for (int kt = 0; kt < Nt; kt++) {
          Vec2 u_xyt = get_Vec2(kt, kx, ky);
          Vec2 u_xyt_filt;

          int count;
          int count_circle = 1;

          if (!std::isnan(u_xyt.f[0]) && !std::isnan(u_xyt.f[1])) {
            u_xyt_filt = u_xyt;
            count      = 1;
          } else {
            u_xyt_filt = Vec2(0.0, 0.0);
            count      = 0;
          }

          for (int kti = -wt; kti <= wt; kti++) {
            for (int kxi = -wx; kxi <= wx; kxi++) {
              for (int kyi = -wy; kyi <= wy; kyi++) {

                float d_circle = std::sqrt(static_cast<float>(kxi * kxi + kyi * kyi));

                if (d_circle <= static_cast<float>(wx)) {
                  count_circle = count_circle + 1;
                }



                if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kt + kti) >= 0 && (kx + kxi) < Nx &&
                    (ky + kyi) < Ny && (kt + kti) < Nt)  // entries must be within array !!!
                {

                  // only non-NaN values are considered
                  if (!std::isnan(displacement(kt + kti, kx + kxi, ky + kyi, 0)) &&
                      !std::isnan(displacement(kt + kti, kx + kxi, ky + kyi, 1))) {

                    float d = std::sqrt(static_cast<float>(kxi * kxi + kyi * kyi));
                    if (d <= static_cast<float>(wx)) {
                      if (kxi == 0 && kyi == 0 && kti == 0) {

                      } else {
                        u_xyt_filt += get_Vec2(kt + kti, kx + kxi, ky + kyi);
                        count = count + 1;
                      }
                    }
                  }
                }
              }
            }
          }

          if (count > wx && count > (count_circle / 4)) {
            set_Vec2_result(kt, kx, ky, u_xyt_filt / static_cast<float>(count));
          } else {
            set_Vec2_result(kt, kx, ky, Vec2(NAN, NAN));
          }
        }
      }
    }
  }
  return result;
}

Array5f spatiotemporalFlowFilter3D(
    const Array5f &displacement, int wx, int wy, int wz, int wt, const Array3f &mask) {
  // treats boundaries differently
  const auto Nt = displacement.shape(0);
  const auto Nx = displacement.shape(1);
  const auto Ny = displacement.shape(2);
  const auto Nz = displacement.shape(3);

  Array5f result(displacement.shape(), std::numeric_limits<float>::quiet_NaN());

  if (displacement.shape(4) != 3) {
    throw std::runtime_error("Error: array should be of type Vec3!");
  }

  const auto test_mask = [&mask](int x, int y, int z) -> bool {
    if (mask.size() > 1) {
      return mask(x, y, z) > 0.5;
    } else {
      return true;
    }
  };

  const auto get_Vec3 = [&displacement](int t, int x, int y, int z) -> Vec3f {
    return Vec3f(displacement(t, x, y, z, 0), displacement(t, x, y, z, 1),
                 displacement(t, x, y, z, 2));
  };

  const auto set_Vec3_result = [&result](int t, int x, int y, int z, Vec3f val) {
    result(t, x, y, z, 0) = val[0];
    result(t, x, y, z, 1) = val[1];
    result(t, x, y, z, 2) = val[2];
  };

  //check whether cumulative sum of deviations is larger than threshold
#pragma omp parallel for collapse(2)
  for (int kx = 0; kx < Nx; kx++) {
    for (int ky = 0; ky < Ny; ky++) {
      for (int kz = 0; kz < Nz; kz++) {
        if (test_mask(kx, ky, kz)) {
          for (int kt = 0; kt < Nt; kt++) {
            Vec3f u_xyt = get_Vec3(kt, kx, ky, kz);
            Vec3f u_xyt_filt;

            int count;
            int count_circle = 1;

            if (!std::isnan(u_xyt[0]) && !std::isnan(u_xyt[1]) && !std::isnan(u_xyt[2])) {
              u_xyt_filt = u_xyt;
              count      = 1;
            } else {
              u_xyt_filt = Vec3f(0.0f, 0.0f, 0.0f);
              count      = 0;
            }

            for (int kti = -wt; kti <= wt; kti++) {
              for (int kxi = -wx; kxi <= wx; kxi++) {
                for (int kyi = -wy; kyi <= wy; kyi++) {
                  for (int kzi = -wy; kzi <= wz; kzi++) {

                    float d_circle =
                        std::sqrt(static_cast<float>(kxi * kxi + kyi * kyi + kzi * kzi));

                    if (d_circle <= static_cast<float>(wx)) {
                      count_circle = count_circle + 1;
                    }

                    if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kz + kzi) >= 0 &&
                        (kt + kti) >= 0 &&  //
                        (kx + kxi) < Nx && (ky + kyi) < Ny && (kz + kzi) < Nz &&
                        (kt + kti) < Nt)  // entries must be within array !!!
                    {

                      // only non-NaN values are considered
                      if (!std::isnan(displacement(kt + kti, kx + kxi, ky + kyi, kz + kzi, 0)) &&
                          !std::isnan(displacement(kt + kti, kx + kxi, ky + kyi, kz + kzi, 1)) &&
                          !std::isnan(displacement(kt + kti, kx + kxi, ky + kyi, kz + kzi, 2))) {

                        float d = std::sqrt(static_cast<float>(kxi * kxi + kyi * kyi + kzi * kzi));
                        if (d <= static_cast<float>(wx)) {
                          if (kxi == 0 && kyi == 0 && kzi == 0 && kti == 0) {

                          } else {
                            u_xyt_filt = u_xyt_filt + get_Vec3(kt + kti, kx + kxi, ky + kyi, kz + kzi);
                            count = count + 1;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (count > wx && count > (count_circle / 6)) {
              set_Vec3_result(kt, kx, ky, kz, u_xyt_filt / static_cast<float>(count));
            } else {
              set_Vec3_result(kt, kx, ky, kz, Vec3f(NAN, NAN, NAN));
            }
          }
        }
      }
    }
  }
  return result;
}