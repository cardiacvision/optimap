#pragma once
#include "xtensor-python/pytensor.hpp"

template <class T, std::size_t N>
using Array   = xt::xtensor<T, N, xt::layout_type::row_major>;
using Array2f = Array<float, 2>;
using Array3f = Array<float, 3>;
using Array4f = Array<float, 4>;
using Array5f = Array<float, 5>;
using Mask2b  = Array<bool, 2>;

template <class T, std::size_t N>
using PyArray   = xt::pytensor<T, N, xt::layout_type::row_major>;
using PyArray2f = PyArray<float, 2>;
using PyArray3f = PyArray<float, 3>;
using PyArray4f = PyArray<float, 4>;
using PyArray5f = PyArray<float, 5>;