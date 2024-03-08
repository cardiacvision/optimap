import numpy as np

import optimap as om


def test_contrast_enhancement():
    img = np.ones((64, 32), dtype=np.uint16)
    vid = np.ones((10, 32, 32), dtype=np.float32)
    out = om.motion.contrast_enhancement(img, 3)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(out == 0)

    mask = np.ones_like(img, dtype=bool)
    out2 = om.motion.contrast_enhancement(img, 3, mask)
    assert np.all(out2 == out)

    out = om.motion.contrast_enhancement(vid, 3)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out == 0)

    mask = np.ones_like(vid, dtype=bool)
    out2 = om.motion.contrast_enhancement(vid, 3, mask)
    assert np.all(out2 == out)

    mask = np.zeros_like(vid, dtype=bool)
    out2 = om.motion.contrast_enhancement(vid, 3, mask)
    assert np.all(out2 == 0)

    mask = mask[0]
    out2 = om.motion.contrast_enhancement(vid, 3, mask)
    assert np.all(out2 == 0)

    vid = np.random.random((10, 32, 32)).astype(np.float32)
    out = om.motion.contrast_enhancement(vid, 3)
    assert out.shape == vid.shape
    assert out.min() == 0
    assert out.max() == 1

    mask = np.random.random((10, 32, 32)) > 0.5
    out = om.motion.contrast_enhancement(vid, 3, mask)
    assert out.shape == vid.shape
    assert out.min() == 0
    assert out.max() == 1

def test_flowestimator():
    estimator = om.motion.FlowEstimator()
    vid = np.ones((10, 128, 128), dtype=np.float32)
    ref = np.ones((128, 128), dtype=np.float32)
    out = estimator.estimate(vid, ref)
    assert out.shape == vid.shape + (2,)
    assert out.dtype == np.float32
    assert np.all(out == 0)

    out = estimator.estimate_reverse(vid, ref)
    assert out.shape == vid.shape + (2,)
    assert out.dtype == np.float32
    assert np.all(out == 0)

def test_warp():
    vid = np.random.random((10, 128, 128)).astype(np.float32)
    flows = np.zeros((10, 128, 128, 2), dtype=np.float32)
    out = om.motion.warp_video(vid, flows)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out == vid)


def test_motion_compensate():
    video = (np.random.random((10, 10, 10)) * 16535).astype(np.uint16)
    warped = om.motion.motion_compensate(video)
    assert warped.shape == video.shape
    assert warped.dtype == video.dtype
