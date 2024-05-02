import numpy as np
import pytest

import optimap as om


def test_normalize():
    vid = (np.random.random((10, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize(vid, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)

    vid = vid.astype(np.float64)
    out = om.video.normalize(vid, dtype="float64")
    assert out.dtype == np.float64
    assert np.allclose(out.max(), 1.0)
    assert np.allclose(out.min(), 0.0)

    out = om.video.normalize(vid, dtype="uint8")
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255

def test_normalize_pixelwise():
    vid = (np.random.random((10, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize_pixelwise(vid, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)

    vid = np.random.random((10, 128, 128))
    vid[:, 0, 0] = np.nan
    vid[2:, 1, 1] = np.nan
    out = om.video.normalize_pixelwise(vid)
    assert np.isnan(out[:, 0, 0]).all()
    assert np.isnan(out[2:, 1, 1]).all()
    out[:, 0, 0] = 1
    out[2:, 1, 1] = 1
    assert not np.isnan(out).any()

def test_normalize_pixelwise_slidingwindow():
    vid = (np.random.random((100, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize_pixelwise_slidingwindow(vid, 21, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5), f"min: {out.min()}, max: {out.max()}"
    assert np.all(out <= 0.5)
    assert np.isnan(out).sum() == 0

    t = 50
    window = vid[t-10:t+11, 50, 50]
    expected = (vid[t, 50, 50] - window.min()) / (window.max() - window.min()) * 1 - 0.5
    assert np.allclose(out[t, 50, 50], expected)

    vid = (np.random.random((10, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize_pixelwise_slidingwindow(vid, 20)
    out2 = om.video.normalize_pixelwise(vid)
    assert np.allclose(out, out2)

def test_temporal_difference():
    video = np.zeros((10, 128, 128), dtype=np.uint16)
    for i in range(1, 10):
        video[i] = i
    out = om.video.temporal_difference(video, 1, fill_value=np.inf)
    assert out.shape == video.shape
    assert out.dtype == np.float32
    assert np.all(out[0] == np.inf)
    assert np.all(out[1:] == 1)

    out = om.video.temporal_difference(video, 2)
    assert np.all(out[2:] == 2)

    out = om.video.temporal_difference(video, 2, fill_value=np.inf, center=True)
    assert out.shape == video.shape
    assert out.dtype == np.float32
    assert np.all(out[0] == np.inf) and np.all(out[-1] == np.inf)
    assert np.all(out[1:-1] == 2)


def test_resize():
    video = np.zeros((10, 128, 128), dtype=np.float32)
    out = om.video.resize(video, shape=(64, 64))
    assert out.shape == (10, 64, 64)
    assert out.dtype == np.float32

    out = om.video.resize(video, scale=0.5)
    assert out.shape == (10, 64, 64)
    assert out.dtype == np.float32

    out = om.video.resize(video, scale=0.5, interpolation="area")
    assert out.shape == (10, 64, 64)

    pytest.raises(ValueError, om.video.resize, video)
    pytest.raises(ValueError, om.video.resize, video, scale=0.5, shape=(64, 64))
    pytest.raises(ValueError, om.video.resize, video, scale=0.5, interpolation="foobar")
