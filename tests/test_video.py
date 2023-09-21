import pytest
import numpy as np

import optimap as om

def test_normalize():
    vid = (np.random.random((10, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize(vid, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)

def test_normalize_pixelwise():
    vid = (np.random.random((10, 128, 128)) * 8000).astype(np.uint16)
    out = om.video.normalize_pixelwise(vid, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)

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
    out = om.video.temporal_difference(video, 1)
    assert out.shape == video.shape
    assert out.dtype == np.float32
    assert np.all(out[0] == 0)
    assert np.all(out[1:] == 1)

    out = om.video.temporal_difference(video, 2)
    assert np.all(out[2:] == 2)
