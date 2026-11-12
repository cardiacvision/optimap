import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import static_ffmpeg.run

import optimap as om
from optimap.video import _ffmpeg

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# lossless codec which can be used to verify the frame contents
LOSSLESS = {"-c:v": "ffv1"}


def _test_video(nt=6, height=4, width=6, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (nt, height, width), dtype=np.uint8)


def _write(filename, frames):
    with _ffmpeg.FFmpegWriter(filename, inputdict={"-r": "30"}, outputdict=LOSSLESS) as writer:
        for frame in frames:
            writer.writeFrame(frame)


def test_gray_roundtrip(tmpdir):
    video = _test_video()
    fn = Path(tmpdir) / "gray.mkv"
    _write(fn, video)
    assert np.array_equal(om.load_video(fn), video)


def test_rgb_roundtrip(tmpdir):
    video = np.stack([_test_video(seed=i) for i in range(3)], axis=-1)
    fn = Path(tmpdir) / "rgb.mkv"
    _write(fn, video)
    loaded = om.load_video(fn, as_grey=False)
    assert loaded.shape == video.shape
    assert np.array_equal(loaded, video)


def test_loaded_video_is_writable(tmpdir):
    fn = Path(tmpdir) / "gray.mkv"
    _write(fn, _test_video())
    video = om.load_video(fn)
    video[0, 0, 0] = 1
    assert video.flags.writeable


def test_frame_selection(tmpdir):
    video = _test_video(nt=10)
    fn = Path(tmpdir) / "gray.mkv"
    _write(fn, video)
    assert np.array_equal(om.load_video(fn, frames=3), video[:3])
    assert np.array_equal(om.load_video(fn, start_frame=2, frames=3), video[2:5])
    assert np.array_equal(om.load_video(fn, start_frame=1, frames=3, step=2), video[1:7:2])
    assert np.array_equal(om.load_video(fn, start_frame=8), video[8:])
    assert np.array_equal(om.load_video(fn, start_frame=-3), video[-3:])
    assert om.load_video(fn, frames=0).shape == (0, 4, 6)


def test_variable_frame_rate_keeps_every_frame(tmpdir):
    """ffmpeg by default duplicates frames to get a constant frame rate when writing raw frames."""
    ffmpeg, _ = _ffmpeg._find_executables()
    fn = Path(tmpdir) / "vfr.mkv"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=10",
                    "-vf", "select='not(mod(n,3))'", "-c:v", "ffv1", str(fn)], check=True)
    assert len(om.load_video(fn)) == 7


def test_rotation_metadata(tmpdir):
    ffmpeg, _ = _ffmpeg._find_executables()
    fn = Path(tmpdir) / "video.mp4"
    rotated = Path(tmpdir) / "rotated.mp4"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
                    "-pix_fmt", "yuv420p", str(fn)], check=True)
    result = subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(fn), "-c", "copy",
                             str(rotated)], capture_output=True)
    if result.returncode != 0:
        pytest.skip("this version of ffmpeg does not support -display_rotation")

    video, video_rotated = om.load_video(fn), om.load_video(rotated)
    assert video.shape == (10, 48, 64)
    assert video_rotated.shape == (10, 64, 48)
    assert np.abs(np.rot90(video[0]).astype(int) - video_rotated[0]).mean() < 2


def test_read_invalid_file(tmpdir):
    fn = Path(tmpdir) / "invalid.mp4"
    fn.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="invalid.mp4"):
        om.load_video(fn)


def test_writer_reports_ffmpeg_errors(tmpdir):
    fn = Path(tmpdir) / "test.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg"):
        with _ffmpeg.FFmpegWriter(fn, outputdict={"-c:v": "doesnotexist"}) as writer:
            writer.writeFrame(np.zeros((4, 4), dtype=np.uint8))

    # odd frame sizes are not supported by libx264 with yuv420p
    with pytest.raises(RuntimeError, match="ffmpeg"):
        with _ffmpeg.FFmpegWriter(fn, outputdict={"-c:v": "libx264", "-pix_fmt": "yuv420p"}) as writer:
            for _ in range(3):
                writer.writeFrame(np.zeros((5, 5), dtype=np.uint8))

    # a large amount of data after ffmpeg exited must raise the ffmpeg error as well, not a BrokenPipeError
    with pytest.raises(RuntimeError, match="ffmpeg"):
        with _ffmpeg.FFmpegWriter(fn, outputdict={"-c:v": "doesnotexist"}) as writer:
            for _ in range(100):
                writer.writeFrame(np.zeros((512, 512), dtype=np.uint8))


def test_writer_validates_frames(tmpdir):
    fn = Path(tmpdir) / "test.mkv"
    writer = _ffmpeg.FFmpegWriter(fn, outputdict=LOSSLESS)
    with pytest.raises(ValueError):
        writer.writeFrame(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        writer.writeFrame(np.zeros((4, 4, 5), dtype=np.uint8))
    writer.writeFrame(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        writer.writeFrame(np.zeros((8, 8), dtype=np.uint8))
    writer.close()
    writer.close()  # closing twice is fine
    assert fn.stat().st_size > 0


def test_writer_without_frames_creates_no_file(tmpdir):
    fn = Path(tmpdir) / "test.mp4"
    with _ffmpeg.FFmpegWriter(fn):
        pass
    assert not fn.exists()


def test_writer_cleans_up_on_exception(tmpdir):
    fn = Path(tmpdir) / "test.mp4"
    writer = _ffmpeg.FFmpegWriter(fn, outputdict={"-c:v": "doesnotexist"})
    with pytest.raises(ValueError, match="boom"):
        with writer:
            writer.writeFrame(np.zeros((4, 4), dtype=np.uint8))
            raise ValueError("boom")
    assert writer._proc is None
    assert writer._stderr is None


def test_static_ffmpeg_fallback(monkeypatch):
    _ffmpeg._find_executables.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(static_ffmpeg.run, "get_or_fetch_platform_executables_else_raise",
                        lambda: ("/static/ffmpeg", "/static/ffprobe"))
    try:
        assert _ffmpeg._find_executables() == ("/static/ffmpeg", "/static/ffprobe")
    finally:
        _ffmpeg._find_executables.cache_clear()
