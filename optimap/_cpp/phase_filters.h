#pragma once
#include <limits>
#include <cmath>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "definitions.h"
#include "vec2.h"

/*
 * Helper functions
 */

Array<Vec2, 3> phase_to_vec(const Array3f &phase) {
  Array<Vec2, 3> u(phase.shape());
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
  for (int kt = 0; kt < phase.shape(0); kt++) {
    for (int kx = 0; kx < phase.shape(1); kx++) {
      for (int ky = 0; ky < phase.shape(2); ky++) {
        float angle = phase(kt, kx, ky);
        if (!std::isnan(angle)) {
          Vec2 unitv    = Vec2(std::cos(angle), std::sin(angle));
          u(kt, kx, ky) = unitv.normalized();
        } else {
          u(kt, kx, ky) = Vec2(NAN, NAN);
        }
      }
    }
  }
  return u;
}

float vec_to_phase(const Vec2 &u) {
  const auto unitv = u.normalized();
  return std::atan2(unitv.f[1], unitv.f[0]);
}

bool test_mask(const Array2f &mask, int x, int y) {
  if (mask.size() > 1) {
    return mask(x, y) > 0.5;
  } else {
    return true;
  }
}

/*
 * Remove outliers in a phase video by comparing them against their neighbors
 */
Array3f filterPhaseAngleThreshold(
    const Array3f &phase, int wx, int wy, int wt, float tr_angle, const Array2f &mask) {
  const auto Nt = phase.shape(0);
  const auto Nx = phase.shape(1);
  const auto Ny = phase.shape(2);
  Array3f result(phase.shape(), std::numeric_limits<float>::quiet_NaN());

  Array<Vec2, 3> u = phase_to_vec(phase);

  //check whether cumulative sum of deviations is larger than threshold
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
  for (int kt = 0; kt < Nt; kt++) {
    for (int kx = 0; kx < Nx; kx++) {
      for (int ky = 0; ky < Ny; ky++) {
        if (test_mask(mask, kx, ky)) {

          Vec2 u_xyt      = u(kt, kx, ky);
          Vec2 u_xyt_mean = u_xyt;
          int count       = 1;

          for (int kti = -wt; kti <= wt; kti++) {
            for (int kxi = -wx; kxi <= wx; kxi++) {
              for (int kyi = -wy; kyi <= wy; kyi++) {

                if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kt + kti) >= 0 && (kx + kxi) < Nx &&
                    (ky + kyi) < Ny && (kt + kti) < Nt) {
                  if (!std::isnan(u(kt + kti, kx + kxi, ky + kyi).f[0])) {

                    float d = std::sqrt(kxi * kxi + kyi * kyi);
                    if (d <= wx) {
                      if (kxi == 0 && kyi == 0 && kti == 0) {

                      } else {
                        u_xyt_mean += u(kt + kti, kx + kxi, ky + kyi);
                        count = count + 1;
                      }
                    }
                  }
                }
              }
            }
          }


          u_xyt_mean = u_xyt_mean / static_cast<float>(count);
          u_xyt_mean = u_xyt_mean.normalized();
          u_xyt      = u_xyt.normalized();

          float angle = std::acos(u_xyt.dot(u_xyt_mean));
          if (angle > tr_angle || std::isnan(u(kt, kx, ky).f[0])) {
            result(kt, kx, ky) = std::numeric_limits<float>::quiet_NaN();
          } else {
            result(kt, kx, ky) = vec_to_phase(u(kt, kx, ky));
          }
        }  // mask
      }
    }
  }

  return result;
}

/*
 *
 */
Array3f filterPhaseDisc(const Array3f &phase, int wx, int wy, int wt, const Array2f &mask) {
  const auto Nt = phase.shape(0);
  const auto Nx = phase.shape(1);
  const auto Ny = phase.shape(2);
  Array3f result(phase.shape(), std::numeric_limits<float>::quiet_NaN());

  Array<Vec2, 3> u = phase_to_vec(phase);

  //check whether cumulative sum of deviations is larger than threshold
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
  for (int kt = 0; kt < Nt; kt++) {
    for (int kx = 0; kx < Nx; kx++) {
      for (int ky = 0; ky < Ny; ky++) {
        if (test_mask(mask, kx, ky)) {
          Vec2 u_xyt_filt = Vec2(0.0, 0.0);
          int count       = 0;

          for (int kti = -wt; kti <= wt; kti++) {
            for (int kxi = -wx; kxi <= wx; kxi++) {
              for (int kyi = -wy; kyi <= wy; kyi++) {
                // clang-format off
                if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kt + kti) >= 0 &&
                    (kx + kxi) < Nx && (ky + kyi) < Ny && (kt + kti) < Nt) {
                  // clang-format on
                  if (!std::isnan(u(kt + kti, kx + kxi, ky + kyi).f[0])) {
                    if (std::sqrt(kxi * kxi + kyi * kyi) <= wx) {
                      if (kxi == 0 && kyi == 0 && kti == 0) {

                      } else {
                        //u_xyt_filt = u_xyt_filt*0.5 + u[kx+kxi][ky+kyi][kt+kti]*0.5;
                        u_xyt_filt += u(kt + kti, kx + kxi, ky + kyi);
                        count = count + 1;
                      }
                    }
                  }
                }
              }
            }
          }

          u_xyt_filt         = u_xyt_filt / static_cast<float>(count);
          result(kt, kx, ky) = vec_to_phase(u_xyt_filt);
        }  // mask
      }
    }
  }

  return result;
}

/*
 * thresh: percentage of NotNaNs in circular kernel
 */
Array3f filterPhaseFillSmooth(
    const Array3f &phase, int wx, int wy, int wt, float thresh, const Array2f &mask) {
  const auto Nt = phase.shape(0);
  const auto Nx = phase.shape(1);
  const auto Ny = phase.shape(2);
  Array3f result(phase.shape(), std::numeric_limits<float>::quiet_NaN());

  Array<Vec2, 3> u = phase_to_vec(phase);

  //check whether cumulative sum of deviations is larger than threshold
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
  for (int kt = 0; kt < Nt; kt++) {
    for (int kx = 0; kx < Nx; kx++) {
      for (int ky = 0; ky < Ny; ky++) {

        if (test_mask(mask, kx, ky)) {
          if (!std::isnan(u(kt, kx, ky).f[0])) {
            result(kt, kx, ky) = phase(kt, kx, ky);
          } else {
            Vec2 u_xyt_filt = Vec2(0.0, 0.0);
            int count       = 0;
            int countNaN    = 1;
            int countNotNaN = 1;

            for (int kti = -wt; kti <= wt; kti++) {
              for (int kxi = -wx; kxi <= wx; kxi++) {
                for (int kyi = -wy; kyi <= wy; kyi++) {
                  // clang-format off
                  if ((kx + kxi) >= 0 && (ky + kyi) >= 0 && (kt + kti) >= 0 &&
                      (kx + kxi) < Nx && (ky + kyi) < Ny && (kt + kti) < Nt) {
                    // clang-format on
                    if (!std::isnan(u(kt + kti, kx + kxi, ky + kyi).f[0])) {
                      countNotNaN += 1;
                      if (std::sqrt(kxi * kxi + kyi * kyi) <= wx) {
                        if (kxi == 0 && kyi == 0 && kti == 0) {

                        } else {
                          //u_xyt_filt = u_xyt_filt*0.5 + u[kx+kxi][ky+kyi][kt+kti]*0.5;
                          u_xyt_filt += u(kt + kti, kx + kxi, ky + kyi);
                          count += 1;
                        }
                      }
                    } else {
                      countNaN += 1;
                    }
                  }
                }
              }
            }

            float percentage = countNotNaN / static_cast<float>(countNotNaN + countNaN);
            if (percentage >= thresh && countNotNaN >= 3) {
              u_xyt_filt         = u_xyt_filt / static_cast<float>(count);
              result(kt, kx, ky) = vec_to_phase(u_xyt_filt);
            } else {
              result(kt, kx, ky) = std::numeric_limits<float>::quiet_NaN();
            }
          }
        }  // mask
      }
    }
  }

  return result;
}