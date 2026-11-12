"""Read and write videos by piping raw frames from and to ffmpeg."""

import functools
import json
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import static_ffmpeg.run

# number of channels of a frame -> ffmpeg pixel format
_PIXEL_FORMATS = {1: "gray", 2: "ya8", 3: "rgb24", 4: "rgba"}


@functools.lru_cache(maxsize=1)
def _find_executables():
    """Return the paths to ffmpeg and ffprobe, static binaries are downloaded if they are not installed."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        ffmpeg, ffprobe = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    return str(ffmpeg), str(ffprobe)


@functools.lru_cache(maxsize=None)
def _passthrough_args(ffmpeg):
    """Return the arguments which output every decoded frame exactly once, without duplicating or dropping frames.

    By default ffmpeg converts variable frame rate videos to a constant frame rate when writing raw frames. ``-vsync``
    was replaced by ``-fps_mode`` in ffmpeg 5.1 and has since been removed.
    """
    help_text = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"], stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, errors="replace", check=False).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in help_text else ["-vsync", "passthrough"]


def _as_args(params):
    """Convert a dictionary of ffmpeg parameters to a list of command line arguments."""
    return [str(arg) for item in params.items() for arg in item]


def _read_log(file):
    file.seek(0)
    return file.read().decode(errors="replace").strip()


def _probe_frame_size(ffprobe, filename):
    """Return (height, width) of the decoded frames of the first video stream in a file."""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
        "-of", "json", str(filename),
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                            check=False)
    streams = json.loads(result.stdout).get("streams", []) if result.returncode == 0 and result.stdout else []
    if not streams:
        msg = f"Unable to find a video stream in file '{filename}': {result.stderr.strip()}"
        raise ValueError(msg)

    stream = streams[0]
    height, width = stream["height"], stream["width"]
    if height <= 0 or width <= 0:
        msg = f"Invalid video frame dimensions {width}x{height} in file '{filename}'"
        raise ValueError(msg)
    # ffmpeg applies the rotation metadata when decoding, which swaps the frame dimensions for 90° rotations
    rotations = [side_data.get("rotation", 0) for side_data in stream.get("side_data_list", [])]
    rotations.append(stream.get("tags", {}).get("rotate", 0))
    if any(int(float(rotation)) % 180 == 90 for rotation in rotations):
        height, width = width, height
    return height, width


def read_video(filename, num_frames=None, as_grey=True):
    """Decode a video file with ffmpeg.

    Parameters
    ----------
    filename : str or pathlib.Path
        Video file to read
    num_frames : int, optional
        Only decode the first ``num_frames`` frames, by default all frames are decoded
    as_grey : bool, optional
        Return the luminance channel instead of RGB frames, by default True

    Returns
    -------
    np.ndarray
        uint8 video of shape (T, H, W) if ``as_grey`` is True, otherwise of shape (T, H, W, 3)
    """
    ffmpeg, ffprobe = _find_executables()
    height, width = _probe_frame_size(ffprobe, filename)
    frame_shape = (height, width) if as_grey else (height, width, 3)
    if num_frames == 0:
        return np.empty((0, *frame_shape), dtype=np.uint8)
    frame_bytes = int(np.prod(frame_shape))

    cmd = [ffmpeg, "-loglevel", "error", "-i", str(filename), "-map", "0:v:0"]
    cmd += _passthrough_args(ffmpeg)
    if num_frames is not None:
        cmd += ["-frames:v", str(num_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray" if as_grey else "rgb24", "-"]

    buffer = bytearray()  # frombuffer() on a bytearray gives a writable array without copying the frames again
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr)
        try:
            while True:
                chunk = proc.stdout.read(2**24)
                if not chunk:
                    break
                buffer += chunk
        except BaseException:
            proc.kill()
            raise
        finally:
            proc.stdout.close()
            returncode = proc.wait()
        log = _read_log(stderr)

    frames = len(buffer) // frame_bytes
    if frames == 0:
        msg = f"No frames could be read from video file '{filename}': {log}"
        raise ValueError(msg)
    if returncode != 0:
        warnings.warn(f"ffmpeg reported an error when reading '{filename}', the video may be incomplete: {log}",
                      UserWarning)
    return np.frombuffer(buffer, dtype=np.uint8, count=frames * frame_bytes).reshape((frames, *frame_shape))


class FFmpegWriter:
    """Encode a video file by piping uint8 frames to ffmpeg.

    The ffmpeg process is started when the first frame is written. Errors reported by ffmpeg are raised as
    :class:`RuntimeError` by :meth:`writeFrame` or :meth:`close`.

    Parameters
    ----------
    filename : str or pathlib.Path
        Video file to write, will be overwritten if it exists
    inputdict : dict, optional
        ffmpeg arguments describing the input, e.g. the framerate ``{"-r": "60"}``
    outputdict : dict, optional
        ffmpeg arguments for the output, e.g. the encoder ``{"-c:v": "libx264"}``
    """

    def __init__(self, filename, inputdict=None, outputdict=None):
        self.filename = str(filename)
        self.inputdict = dict(inputdict or {})
        self.outputdict = dict(outputdict or {})
        self._proc = None
        self._stderr = None
        self._frame_shape = None

    def _start(self, frame):
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        if channels not in _PIXEL_FORMATS:
            msg = f"Frames must have one of {sorted(_PIXEL_FORMATS)} channels, got frame of shape {frame.shape}"
            raise ValueError(msg)

        ffmpeg, _ = _find_executables()
        height, width = frame.shape[:2]
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", _PIXEL_FORMATS[channels],
               "-s", f"{width}x{height}"]
        cmd += _as_args(self.inputdict)
        cmd += ["-i", "-"]
        cmd += _as_args(self.outputdict)
        cmd.append(self.filename)

        self._frame_shape = frame.shape
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)

    def writeFrame(self, frame):  # noqa: N802
        """Write a frame of shape (H, W) or (H, W, C) and dtype uint8 with C=1 gray, C=3 RGB, or C=4 RGBA."""
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            msg = f"Frames must be of type uint8, got {frame.dtype}"
            raise ValueError(msg)
        if frame.ndim not in (2, 3):
            msg = f"Frames must have shape (H, W) or (H, W, C), got {frame.shape}"
            raise ValueError(msg)
        if self._proc is None:
            self._start(frame)
        elif frame.shape != self._frame_shape:
            msg = f"All frames must have the same shape, expected {self._frame_shape}, got {frame.shape}"
            raise ValueError(msg)

        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.close()  # raises the error reported by ffmpeg
            raise RuntimeError(f"ffmpeg stopped while writing '{self.filename}'") from None

    def _close(self, check=True):
        if self._proc is None:
            return
        proc, stderr = self._proc, self._stderr
        self._proc = self._stderr = None
        try:
            proc.stdin.close()
        except BrokenPipeError:  # remaining buffered data could not be flushed, ffmpeg already exited
            pass
        returncode = proc.wait()
        log = _read_log(stderr)
        stderr.close()
        if check and returncode != 0:
            msg = f"ffmpeg failed to write '{self.filename}' (exit code {returncode}): {log}"
            raise RuntimeError(msg)

    def close(self):
        """Finish encoding the video, raises a :class:`RuntimeError` if ffmpeg reported an error."""
        self._close(check=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc_info):
        # Preserve an active exception rather than masking it if ffmpeg also exits with an error
        self._close(check=exc_type is None)
