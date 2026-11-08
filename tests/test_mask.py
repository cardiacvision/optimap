import matplotlib.pyplot as plt
import numpy as np
import pytest

import optimap as om


def test_background_threshold_nan_and_float_range():
    rng = np.random.default_rng(0)
    img = np.concatenate([rng.normal(0.2, 0.05, (50, 100)), rng.normal(0.8, 0.05, (50, 100))]).astype(np.float32)
    thr = om.image.detect_background_threshold(img)
    assert 0.3 < thr < 0.7

    img_nan = img.copy()
    img_nan[0, :10] = np.nan
    assert om.image.detect_background_threshold(img_nan) == pytest.approx(thr, abs=0.05)

    # float values in range (2, 5) used to raise NotImplementedError
    assert 0.9 < om.image.detect_background_threshold(img * 3) < 2.1
    # negative values
    assert -0.5 < om.image.detect_background_threshold(img - 0.5) < 0.2

    mask = om.foreground_mask(img_nan, show=False)
    assert mask[75:].mean() > 0.95 and mask[:50].mean() < 0.05


def test_background_threshold_int16():
    rng = np.random.default_rng(0)
    img = np.concatenate([rng.normal(200, 20, (50, 100)), rng.normal(800, 20, (50, 100))]).astype(np.int16)
    thr = om.image.detect_background_threshold(img)
    assert isinstance(thr, int)
    assert 300 < thr < 700


def test_largest_mask_component_empty():
    mask = np.zeros((10, 10), dtype=bool)
    assert not om.image.largest_mask_component(mask, show=False).any()
