from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import optimap as om


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
