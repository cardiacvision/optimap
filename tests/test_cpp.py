import numpy as np
import pytest

import optimap._cpp as cpp


def test_package():
    val = cpp.is_openmp_enabled()
    assert isinstance(val, bool)

def test_filterPhaseAngleThreshold():
    phase = np.ones((100, 100, 100))
    mask = np.ones((100, 100))
    phase[:, 10, 10] = -1
    result = cpp.phasefilter_angle_threshold(phase, 2, 2, 2, 1, mask)
    assert np.isnan(result[:, 10, 10]).all()
    result[:, 10, 10] = 1
    assert np.all(result == 1)

def test_filterPhaseDisc():
    phase = np.ones((100, 100, 100))
    mask = np.ones((100, 100))
    phase[:, 10, 10] = np.nan
    result = cpp.phasefilter_disc(phase, 3, 3, 0, mask)
    assert np.all(result > 0.99)

def test_filterPhaseFillSmooth():
    phase = np.ones((100, 100, 100))
    mask = np.ones((100, 100))
    phase[:, 10, 10] = np.nan
    result = cpp.phasefilter_fillsmooth(phase, 2, 2, 0, 0.4, mask)
    assert np.all(result == 1)

def test_contrastEnhanceImg():
    img = np.ones((64, 32), dtype=np.uint16)
    out = cpp.contrast_enhancement_img(img, 3)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(out == 0)

    pytest.raises(RuntimeError, cpp.contrast_enhancement_img, img, 2)
    
    mask = np.ones((64, 32), dtype=bool)
    out = cpp.contrast_enhancement_img(img, 3, mask)

def test_contrastEnhanceVideo():
    vid = np.ones((100, 64, 32), dtype=np.uint16)
    out = cpp.contrast_enhancement_video(vid, 3)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out == 0)

    pytest.raises(RuntimeError, cpp.contrast_enhancement_video, vid, 4)

def test_spatiotemporalFlowFilter():
    vid = np.ones((100, 64, 32, 2), dtype=np.float32)
    out = cpp.flowfilter_smooth_spatiotemporal(vid, 1, 1, 1)
    assert out.shape == vid.shape
    assert out.dtype == vid.dtype
    assert np.all(vid == 1)
