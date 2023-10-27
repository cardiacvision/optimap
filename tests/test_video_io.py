from pathlib import Path
import mimetypes

import pytest
import numpy as np
import optimap as om
import skvideo


def test_npy_video(tmpdir):
    vid = np.random.random((8, 4, 6)).astype(np.float32)

    filename = tmpdir / "test.npy"
    om.save_video(vid, filename)
    assert Path(filename).exists()

    video = om.load_video(filename)
    assert video.shape == vid.shape
    assert video.dtype == vid.dtype
    assert np.all(video == vid)

    video = om.load_video(filename, start_frame=4)
    assert video.shape == (4, 4, 6)
    assert np.all(video == vid[4:])

    video = om.load_video(filename, frames=4)
    assert video.shape == (4, 4, 6)
    assert np.all(video == vid[:4])

    video = om.load_video(filename, step=2)
    assert video.shape == (4, 4, 6)
    assert np.all(video == vid[::2])

    video = om.load_video(filename, use_mmap=True)
    assert np.all(video == vid)
    assert video.flags["WRITEABLE"] == False


def test_tiff_folder(tmpdir):
    vid = np.random.random((10, 4, 6)).astype(np.float32)

    om.save_image_sequence(vid, directory=tmpdir, suffix=".tiff")
    assert len(list(Path(tmpdir).glob("*.tiff"))) == 10

    video = om.load_video(tmpdir)
    assert video.shape == vid.shape
    assert video.dtype == vid.dtype
    assert np.all(video == vid)

    video = om.load_video(tmpdir, start_frame=1, frames=3, step=2)
    assert video.shape == (3, 4, 6)
    assert np.all(video == vid[1:6:2])


def test_png_folder(tmpdir):
    vid = np.random.random((10, 4, 6)) * 16535
    vid = vid.astype(np.uint16)

    om.save_image_sequence(vid, directory=tmpdir, suffix=".png")
    assert len(list(Path(tmpdir).glob("*.png"))) == 10

    video = om.load_video(tmpdir)
    assert video.shape == vid.shape
    assert video.dtype == vid.dtype
    assert np.all(video == vid)

    video = om.load_video(tmpdir, start_frame=1, frames=3, step=2)
    assert video.shape == (3, 4, 6)
    assert np.all(video == vid[1:6:2])


def test_tiff_stack(tmpdir):
    vid = np.random.random((10, 4, 6)).astype(np.float32)
    filename = tmpdir / "test.tiff"
    om.save_video(vid, filename)
    assert Path(filename).exists()
    assert Path(filename).is_file()

    video = om.load_video(filename)
    assert video.shape == vid.shape
    assert video.dtype == vid.dtype
    assert np.all(video == vid)

    video = om.load_video(filename, start_frame=1, frames=3, step=2, use_mmap=True)
    assert video.shape == (3, 4, 6)
    assert np.all(video == vid[1:6:2])
    assert video.flags["WRITEABLE"] == False


def test_matlab(tmpdir):
    vid = np.random.random((10, 4, 6)).astype(np.float32)

    filename = tmpdir / "test.mat"
    om.save_video(vid, filename)
    assert Path(filename).exists()
    assert Path(filename).is_file()

    video = om.load_video(filename)
    assert video.shape == vid.shape
    assert video.dtype == vid.dtype
    assert np.all(video == vid)

    video = om.load_video(filename, start_frame=1, frames=3, step=2)
    assert video.shape == (3, 4, 6)
    assert np.all(video == vid[1:6:2])


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


@pytest.mark.skipif(skvideo._HAS_FFMPEG == 0, reason="ffmpeg not installed")
def test_export_video(tmpdir):
    video = np.random.random((10, 4, 6)).astype(np.float32)
    filename = tmpdir / "test.mp4"
    om.export_video(video, filename, vmin=0, vmax=1)
    assert Path(filename).exists()
    assert Path(filename).is_file()
    assert mimetypes.guess_type(filename)[0] == "video/mp4"


@pytest.mark.skipif(skvideo._HAS_FFMPEG == 0, reason="ffmpeg not installed")
def test_export_video_overlay(tmpdir):
    video = np.random.random((10, 4, 4)).astype(np.float32)
    overlay = np.random.random((10, 4, 4)).astype(np.float32)
    filename = tmpdir / "test.mp4"
    om.video.export_video_with_overlay(video, filename=filename, overlay=overlay, vmin_base=0, vmax_base=1)
    assert Path(filename).is_file()
    assert mimetypes.guess_type(filename)[0] == "video/mp4"

    om.video.export_video_with_overlay(video, filename=filename, overlay=overlay, alpha=overlay)
    assert Path(filename).is_file()
    assert mimetypes.guess_type(filename)[0] == "video/mp4"
