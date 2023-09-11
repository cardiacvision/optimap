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

    out = om.video.normalize_pixelwise(vid, -0.5, 0.5)
    assert out.shape == vid.shape
    assert out.dtype == np.float32
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)

def test_sliding_window():
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


def test_alpha_blending():
    base = np.full((10, 128, 128), 0, dtype=np.float32)
    overlay = np.full((10, 128, 128), 1, dtype=np.float32)
    out = om.video.alpha_blend_videos(base, overlay, alpha=base)
    assert out.shape == base.shape + (4,)
    assert np.allclose(out[..., :3], 0)

    out = om.video.alpha_blend_videos(base, overlay, alpha=overlay, vmin_base=0, vmax_base=1, vmin_overlay=0, vmax_overlay=1, cmap_overlay='gray')
    assert np.allclose(out[..., :3], 1)

    overlay = np.full((10, 128, 128, 4), 1, dtype=np.float32)
    out = om.video.alpha_blend_videos(base, overlay, cmap_overlay='gray')
    assert np.allclose(out[..., :3], 1)

    out = om.video.alpha_blend_videos(base, overlay, alpha=overlay)
    assert np.allclose(out[..., :3], 1)

    out = om.video.alpha_blend_videos(base, overlay, alpha=overlay[0, :, :, 0], cmap_overlay='gray')
    assert np.allclose(out[..., :3], 1)

def test_ffmpeg_defaults():
    pytest.raises(ValueError, om.video.set_default_ffmpeg_encoder, 'doesnotexist')
    om.video.set_default_ffmpeg_encoder('h264_nvenc')
    assert om.video.get_default_ffmpeg_encoder() == 'h264_nvenc'

    om.video.set_ffmpeg_defaults('h264_nvenc', {'-preset': 'slow'})
    om.video.set_default_ffmpeg_encoder('libx264')