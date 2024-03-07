#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>

#define FORCE_IMPORT_ARRAY  // for xtensor
#include "contrast_enhancement.h"
#include "filters.h"
#include "phase_filters.h"
#include "flow_filters.h"

#ifdef USE_OMP
#include <omp.h>
#endif

/*
 * Check if compiled with openmp and that it works
 */
bool is_openmp_enabled() {
#ifdef USE_OMP
  return omp_get_max_threads() > 1;
#else
  return false;
#endif
}

/* contrast enhancement */
PyArray2f py_contrast_enhancement_img(const PyArray2f &img,
                                      std::size_t kernel_size,
                                      const PyArray2b &mask,
                                      std::string kernel_type) {
  return contrast_enhancement_img(img, kernel_size, mask, kernel_type);
}

PyArray3f py_contrast_enhancement_video(const PyArray3f &video,
                                        std::size_t kernel_size,
                                        const PyArray3b &mask,
                                        std::string kernel_type) {
  return contrast_enhancement_video(video, kernel_size, mask, kernel_type);
}

/* regular filters */
PyArray3f py_normalizePixelwiseSlidingWindow(const PyArray3f &video,
                                             int wt,
                                             float ymin,
                                             float ymax) {
  return normalizePixelwiseSlidingWindow(video, wt, ymin, ymax);
}

/* phase filters */
PyArray3f py_filterPhaseAngleThreshold(
    const PyArray3f &phase, int wx, int wy, int wt, float tr_angle, PyArray2f mask) {
  return filterPhaseAngleThreshold(phase, wx, wy, wt, tr_angle, mask);
}
PyArray3f py_filterPhaseDisc(const PyArray3f &phase, int wx, int wy, int wt, PyArray2f mask) {
  return filterPhaseDisc(phase, wx, wy, wt, mask);
}
PyArray3f py_filterPhaseFillSmooth(
    const PyArray3f &phase, int wx, int wy, int wt, float thresh, PyArray2f mask) {
  return filterPhaseFillSmooth(phase, wx, wy, wt, thresh, mask);
}

/* flow filters */
PyArray4f py_spatiotemporalFlowFilter(
    const PyArray4f &displacement, int wx, int wy, int wt, const PyArray2f &mask) {
  return spatiotemporalFlowFilter(displacement, wx, wy, wt, mask);
}
PyArray5f py_spatiotemporalFlowFilter3D(
    const PyArray5f &displacement, int wx, int wy, int wz, int wt, const PyArray3f &mask) {
  return spatiotemporalFlowFilter3D(displacement, wx, wy, wz, wt, mask);
}

PYBIND11_MODULE(_cpp, m) {
  xt::import_numpy();
  using namespace pybind11::literals;

  m.doc() = R"doc(
        Optimap C++ core functions
        -----------------------
        .. currentmodule:: _cpp
        .. autosummary::
           :toctree: _generate
           is_openmp_enabled
           contrast_enhancement_img
           contrast_enhancement_video
           normalizePixelwiseSlidingWindow
           filterPhaseAngleThreshold
           filterPhaseDisc
           filterPhaseFillSmooth
           spatiotemporalFlowFilter
    )doc";

  m.def("is_openmp_enabled", is_openmp_enabled, "");

  m.def("contrast_enhancement_img", py_contrast_enhancement_img, "img"_a, "kernel_size"_a,
        "mask"_a = PyArray2b(), "kernel_type"_a = std::string("circle"));
  m.def("contrast_enhancement_video", py_contrast_enhancement_video, "video"_a, "kernel_size"_a,
        "mask"_a = PyArray3b(), "kernel_type"_a = std::string("circle"));

  m.def("normalize_pixelwise_slidingwindow", py_normalizePixelwiseSlidingWindow, "video"_a, "wt"_a,
        "ymin"_a = 0, "ymax"_a = 1);

  m.def("phasefilter_angle_threshold", py_filterPhaseAngleThreshold,
        "Remove outliers in a phase video by comparing them against their neighbors", "phase"_a,
        "wx"_a, "wy"_a, "wt"_a, "tr_angle"_a, "mask"_a = PyArray2f());
  m.def("phasefilter_disc", py_filterPhaseDisc, "phase"_a, "wx"_a, "wy"_a, "wt"_a,
        "mask"_a = PyArray2f());
  m.def("phasefilter_fillsmooth", py_filterPhaseFillSmooth, "phase"_a, "wx"_a, "wy"_a, "wt"_a,
        "thresh"_a, "mask"_a = PyArray2f());

  m.def("flowfilter_smooth_spatiotemporal", py_spatiotemporalFlowFilter, "displacement"_a, "wx"_a,
        "wy"_a, "wt"_a, "mask"_a = PyArray2f());
  m.def("flowfilter_smooth_spatiotemporal3D", py_spatiotemporalFlowFilter3D, "displacement"_a,
        "wx"_a, "wy"_a, "wz"_a, "wt"_a, "mask"_a = PyArray3f());
}
