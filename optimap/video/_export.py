import os
import warnings
from pathlib import Path
from typing import Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import skvideo
import skvideo.io
import static_ffmpeg.run
from matplotlib.colors import Normalize, to_rgba
from scipy.special import comb
from tqdm import tqdm

from ..image import collage as collage_images
from ..utils import deprecated

FFMEG_DEFAULTS = {
    "libx264": {
        "-c:v": "libx264",
        "-crf": "15",
        "-preset": "slower",
        "-pix_fmt": "yuv420p",
    },
    "h264_nvenc": {
        "-c:v": "h264_nvenc",
        "-preset": "slow",
        "-profile:v": "high",
        "-rc": "vbr_hq",
        "-qmin:v": "19",
        "-qmax:v": "21",
        "-pix_fmt": "yuv420p",
    },
}
DEFAULT_FFMPEG_ENCODER = "libx264"


def _fix_ffmpeg_location():
    """Make skvideo use static ffmpeg if system ffmpeg not found."""
    if not skvideo._HAS_FFMPEG:
        # ffmpeg not found, download static binary for it
        ffmpeg, _  = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        skvideo.setFFmpegPath(str(Path(ffmpeg).parent))
        assert skvideo._HAS_FFMPEG

        # fix global variables
        skvideo.io.ffmpeg._HAS_FFMPEG = skvideo._HAS_FFMPEG
        skvideo.io.ffmpeg._FFMPEG_PATH = skvideo._FFMPEG_PATH
        skvideo.io.ffmpeg._FFMPEG_SUPPORTED_DECODERS = skvideo._FFMPEG_SUPPORTED_DECODERS
        skvideo.io.ffmpeg._FFMPEG_SUPPORTED_ENCODERS = skvideo._FFMPEG_SUPPORTED_ENCODERS


def _ffmpeg_defaults(encoder: str = "libx264"):
    if encoder not in FFMEG_DEFAULTS:
        msg = f"ffmpeg encoder {encoder} not supported"
        raise ValueError(msg)
    else:
        return FFMEG_DEFAULTS[encoder]


def set_default_ffmpeg_encoder(encoder: str):
    """Set the default ffmpeg encoder to use for exporting videos.

    Parameters
    ----------
    encoder : str
        The ffmpeg encoder to use. E.g. 'libx264' or 'h264_nvenc'.
    """
    global DEFAULT_FFMPEG_ENCODER
    if encoder not in FFMEG_DEFAULTS:
        msg = f"Encoder {encoder} unknown, add it first using set_ffmpeg_defaults()"
        raise ValueError(
            msg
        )
    DEFAULT_FFMPEG_ENCODER = encoder


def get_default_ffmpeg_encoder():
    """Get the default ffmpeg encoder to use for exporting videos.

    Returns
    -------
    str
        The ffmpeg encoder to use. E.g. 'libx264' or 'h264_nvenc'.
    """
    return DEFAULT_FFMPEG_ENCODER


def set_ffmpeg_defaults(encoder: str, params: dict):
    """Set the default ffmpeg parameters to use for exporting videos.

    Parameters
    ----------
    encoder : str
        The ffmpeg encoder for which the parameters apply. E.g. 'libx264' or 'h264_nvenc'.
    params : dict
        The ffmpeg parameters to use. E.g. {'-c:v': 'libx264', '-crf': '15', '-preset': 'slower'}.
    """
    FFMEG_DEFAULTS[encoder] = params


class FFmpegWriter(skvideo.io.FFmpegWriter):
    """Wrapper around `skvideo.io.FFmpegWriter` which downloads static binaries if ffmpeg is not installed."""

    def __init__(self, *args, **kwargs):
        _fix_ffmpeg_location()

        super().__init__(*args, **kwargs)


def export_video(
    filename: Union[str, Path],
    video: Union[np.ndarray, Iterable[np.ndarray]],
    fps: int = 60,
    skip_frames: int = 1,
    cmap = "gray",
    vmin : float = None,
    vmax : float = None,
    progress_bar : bool = True,
    ffmpeg_encoder : str = None,
    pad_mode : str = "edge",
):
    """Export a video numpy array to a video file (e.g. ``.mp4``) using `ffmpeg <https://www.ffmpeg.org>`_.

    Downloads pre-built ffmpeg automatically if ffmpeg is not installed.

    Example
    -------
    .. code-block:: python

        import optimap as om
        import numpy as np

        video = np.random.rand(100, 100, 100)
        om.export_video("video.mp4", video, fps=30, cmap="viridis")

    Parameters
    ----------
    filename : str or Path
        Video file path for writing.
    video : np.ndarray or Iterable[np.ndarray]
        The video to export. Should be of shape (frames, height, width, channels) or (frames, height, width).
        If the video is grayscale, the colormap will be applied. If it's an RGB video, its values should range
        [0, 1] or [0, 255] (np.uint8).
    fps : int, optional
        The framerate of the output video, by default 60
    skip_frames : int, optional
        Only export every ``skip_frames`` frames, by default 1
    cmap : str, optional
        The matplotlib colormap to use, by default "gray"
    vmin : float, optional
        The minimum value for the colormap, by default None
    vmax : float, optional
        The maximum value for the colormap, by default None
    progress_bar : bool, optional
        Whether to show a progress bar, by default True
    ffmpeg_encoder : str, optional
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder` and
        :func:`set_ffmpeg_defaults` for more information.
    pad_mode : str, optional
        Odd-sized videos need to be padded to even dimensions, otherwise ffmpeg will error. The padding mode to use,
        by default "edge". See :func:`numpy.pad` for more information.
    """
    if isinstance(video, (str, os.PathLike)):
        filename, video = video, filename
        warnings.warn("WARNING: The order of arguments for optimap.export_video() has changed. "
                      "Please use export_video(filename, video, ...) instead of export_video(video, filename, ...).",
                      DeprecationWarning)
    if ffmpeg_encoder is None:
        ffmpeg_encoder = DEFAULT_FFMPEG_ENCODER

    if isinstance(video, np.ndarray) and (video.ndim < 3 or video.ndim > 4):
        raise ValueError(f"videos has invalid shape {video.shape}, expected (T, X, Y) or (T, X, Y, C)")

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    if skip_frames != 1:
        video = video[::skip_frames]

    writer = FFmpegWriter(
        filename,
        inputdict={"-r": f"{fps}"},
        outputdict=_ffmpeg_defaults(ffmpeg_encoder),
    )

    for frame in tqdm(video, desc="exporting video", disable=not progress_bar):
        if frame.ndim == 2:
            frame = cmap(norm(frame))

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # pad odd-sized frames to even dimension, otherwise ffmpeg will error
        if frame.shape[0] % 2 == 1:
            frame = np.pad(frame, ((0, 1), (0, 0), (0, 0)), mode=pad_mode)
        if frame.shape[1] % 2 == 1:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), mode=pad_mode)

        writer.writeFrame(frame)
    writer.close()
    print(f"video exported to {filename}")


def export_videos(
        filename: Union[str, Path],
        videos: Union[np.ndarray, List[np.ndarray]],
        ncols: int = 6,
        padding: int = 0,
        padding_color = "black",
        fps: int = 60,
        skip_frames: int = 1,
        cmaps: Union[str, List[str]] = "gray",
        vmins : Union[float, List[float]] = None,
        vmaxs : Union[float, List[float]] = None,
        ffmpeg_encoder : str = None,
        progress_bar : bool = True,
):
    """Export a list of videos to a video file (e.g. ``.mp4``) by arranging them side by side in a grid.

    Uses :func:`optimap.image.collage` to arrange the video frames in a grid.

    Example
    -------

    .. code-block:: python

        import optimap as om
        import numpy as np

        videos = [
            np.random.rand(100, 100, 100),
            np.random.rand(100, 100, 100),
            np.random.rand(100, 100, 100),
            np.random.rand(100, 100, 100),
        ]

        cmaps = ["gray", "viridis", "plasma", "inferno"]
        om.export_videos("collage.mp4", videos, ncols=2, padding=10,
                                padding_color="white", cmaps=cmaps)

    Parameters
    ----------
    filename : str or Path
        Video file path for writing.
    videos : np.ndarray or List[np.ndarray]
        The videos to export. Should be of shape (frames, height, width, channels) or (frames, height, width).
        If the video is grayscale, the colormap will be applied. If it's an RGB video, its values should range
        [0, 1] or [0, 255] (np.uint8).
    ncols : int, optional
        The number of columns in the collage, by default 6
    padding : int, optional
        The padding between the videos in pixels, by default 0
    padding_color : str or np.ndarray (np.uint8), optional
        The color of the padding, by default 'black'
    fps : int, optional
        The framerate of the output video, by default 60
    skip_frames : int, optional
        Only export every ``skip_frames`` frame, by default 1
    cmaps : str or List[str], optional
        The matplotlib colormap to use, by default "gray"
    vmins : float or List[float], optional
        The minimum value for the colormap, by default None
    vmaxs : float or List[float], optional
        The maximum value for the colormap, by default None
    ffmpeg_encoder : str, optional
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder` and
        :func:`set_ffmpeg_defaults` for more information.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True
    """
    if isinstance(videos, (str, os.PathLike)):
        filename, videos = videos, filename
        warnings.warn("WARNING: The order of arguments for optimap.export_videos() has changed. "
                      "Please use export_videos(filename, videos, ...) instead of "
                      "export_videos(videos, filename, ...).",
                      DeprecationWarning)
    if ffmpeg_encoder is None:
        ffmpeg_encoder = DEFAULT_FFMPEG_ENCODER

    if isinstance(videos, np.ndarray) and (videos.ndim < 3 or videos.ndim > 4):
        raise ValueError(f"videos has invalid shape {videos.shape}, expected (N, T, X, Y) or (N, T, X, Y, C)")
    else:
        for video in videos:
            if video.ndim < 3 or video.ndim > 4:
                raise ValueError(f"videos has invalid shape {video.shape}, expected (T, X, Y) or (T, X, Y, C)")

    Nt = len(videos[0])
    for video in videos[1:]:
        if len(video) != Nt:
            raise ValueError("all videos must have the same number of frames")

    if isinstance(cmaps, str):
        cmaps = plt.get_cmap(cmaps)
    if not isinstance(cmaps, list):
        cmaps = [cmaps] * len(videos)
    else:
        for i in range(len(cmaps)):
            if isinstance(cmaps[i], str):
                cmaps[i] = plt.get_cmap(cmaps[i])

    if not isinstance(vmins, list):
        vmins = [vmins] * len(videos)
    if not isinstance(vmaxs, list):
        vmaxs = [vmaxs] * len(videos)

    if len(videos) != len(cmaps) != len(vmins) != len(vmaxs):
        raise ValueError("videos, cmaps, vmins, and vmaxs must have the same length")

    def transform_frame(frame, cmap, vmin, vmax):
        if frame.ndim == 2:
            frame = cmap(Normalize(vmin=vmin, vmax=vmax, clip=True)(frame))
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        return frame

    if isinstance(padding_color, str):
        padding_color = np.array(to_rgba(padding_color))
        padding_color = (padding_color * 255).astype(np.uint8)

    writer = FFmpegWriter(
        filename,
        inputdict={"-r": f"{fps}"},
        outputdict=_ffmpeg_defaults(ffmpeg_encoder),
    )
    for t in tqdm(range(0, Nt, skip_frames), desc="exporting video", disable=not progress_bar):
        frames = [transform_frame(video[t], cmap, vmin, vmax)
                  for video, cmap, vmin, vmax in zip(videos, cmaps, vmins, vmaxs)]
        frame = collage_images(frames, padding=padding, padding_value=padding_color, ncols=ncols)
        writer.writeFrame(frame)
    writer.close()
    print(f"Video exported to {filename}")


@deprecated("Use export_videos() instead")
def export_video_collage(*args, **kwargs):
    return export_videos(*args, **kwargs)


def smoothstep(x, vmin=0, vmax=1, N=2):
    """Smoothly clamps the input array to the range [0, 1] using the `smoothstep function <https://en.wikipedia.org/wiki/Smoothstep>`_.
    Useful for e.g. creating alpha channels.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    vmin : float, optional
        The minimum value of the input array, by default 0
    vmax : float, optional
        The maximum value of the input array, by default 1
    N : int, optional
        The order of the polynomial, by default 2

    Returns
    -------
    np.ndarray
        The smoothly clamped array, with values in the range [0, 1].
    """
    x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def iter_alpha_blend_videos(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: np.ndarray = None,
    skip_frames: int = 1,
    cmap_base="gray",
    cmap_overlay="Purples",
    vmin_base=None,
    vmax_base=None,
    vmin_overlay=None,
    vmax_overlay=None,
):
    """Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_. Yields the
    blended video frame by frame.

    The base video is blended with the overlay video using an alpha channel. If no alpha channel is provided, the
    overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used
    (if it is RGBA). The alpha channel is expected to be in the range [0, 1], a value of 0 means that the base video
    is used, a value of 1 means that the overlay video is used.

    Parameters
    ----------
    base : np.ndarray
        The base video. Should be of shape (frames, height, width) or (frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    overlay : np.ndarray
        The overlay video. Should be of shape (frames, height, width) or(frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    alpha : np.ndarray, optional
        The alpha channel to use for blending, by default ``None``. Expected to be in range [0, 1], and of shape
        (T, X, Y) or (X, Y). If ``None``, the overlay array is used (if grayscale) or the alpha channel of the
        overlay video is used (if RGBA).
    skip_frames : int, optional
        Only export every ``skip_frames`` frames, by default 1
    cmap_base : str or matplotlib.colors.Colormap, optional
        The colormap to use for the base video, by default ``'gray'``
    cmap_overlay : str or matplotlib.colors.Colormap, optional
        The colormap to use for the overlay video, by default ``'Purples'``
    vmin_base : float, optional
        The minimum value for the base video, by default None
    vmax_base : float, optional
        The maximum value for the base video, by default None
    vmin_overlay : float, optional
        The minimum value for the overlay video, by default None
    vmax_overlay : float, optional
        The maximum value for the overlay video, by default None
    ffmpeg_encoder : str, optional
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder`
        and :func:`set_ffmpeg_defaults` for more information.

    Yields
    ------
    np.ndarray {height, width, 4}
        The blended video frame.
    """
    # Create alpha channel with shape (frames, height, width, 1)
    if alpha is None:
        if overlay.ndim == 4:
            alpha = overlay[..., 3, np.newaxis]
        elif overlay.ndim == 3:
            alpha = overlay[..., np.newaxis]
    elif isinstance(alpha, (int, float)):
        alpha = np.full(base.shape[1, 2], alpha)[np.newaxis, :, :, np.newaxis]
        alpha = np.repeat(alpha, base.shape[0], axis=0)
    else:
        if alpha.ndim == 2:
            alpha = alpha[np.newaxis, :, :, np.newaxis]
            alpha = np.repeat(alpha, overlay.shape[0], axis=0)
        elif alpha.ndim == 3:
            alpha = alpha[..., np.newaxis]

    if overlay.ndim == 4 and overlay.shape[-1] == 4:
        nans = np.isnan(overlay[..., 3])
        if np.any(nans):
            alpha = np.copy(alpha)
            alpha[nans] = 0
    else:
        nans = np.isnan(overlay)
        if np.any(nans):
            alpha = np.copy(alpha)
            alpha[nans] = 0

    if isinstance(cmap_base, str):
        cmap_base = plt.get_cmap(cmap_base)
    if isinstance(cmap_overlay, str):
        cmap_overlay = plt.get_cmap(cmap_overlay)

    if vmin_base is None:
        vmin_base = np.nanmin(base[0])
    if vmax_base is None:
        vmax_base = np.nanmax(base[0])
    if vmin_overlay is None:
        vmin_overlay = np.nanmin(overlay)
    if vmax_overlay is None:
        vmax_overlay = np.nanmax(overlay)

    norm1 = Normalize(vmin=vmin_base, vmax=vmax_base, clip=True)
    norm2 = Normalize(vmin=vmin_overlay, vmax=vmax_overlay, clip=True)

    for f_base, f_overlay, f_alpha in zip(
        base[::skip_frames],
        overlay[::skip_frames],
        alpha[::skip_frames],
    ):
        if f_base.ndim == 2:
            f_base = cmap_base(norm1(f_base))
        if f_overlay.ndim == 2:
            f_overlay = cmap_overlay(norm2(f_overlay))
        f_alpha = np.clip(f_alpha, 0, 1)

        frame = f_base * (1 - f_alpha) + f_overlay * f_alpha
        frame = np.clip(frame, 0, 1)  # sometimes rounding errors lead to values > 1
        yield frame


def export_video_with_overlay(
    filename: Union[str, Path],
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: np.ndarray = None,
    fps: int = 60,
    skip_frames: int = 1,
    cmap_base="gray",
    cmap_overlay="Purples",
    vmin_base=None,
    vmax_base=None,
    vmin_overlay=None,
    vmax_overlay=None,
    ffmpeg_encoder=None,
    progress_bar : bool = True,
):
    """Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_ and
    exports it to a video file (e.g. ``.mp4``).

    The base video is blended with the overlay video using an alpha channel. If no alpha channel is provided, the
    overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used
    (if it is RGBA). The alpha channel is expected to be in the range [0, 1], a value of 0 means that the base video
    is used, a value of 1 means that the overlay video is used.

    Parameters
    ----------
    filename : str or Path
        Video file path for writing.
    base : np.ndarray
        The base video. Should be of shape (frames, height, width) or (frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    overlay : np.ndarray
        The overlay video. Should be of shape (frames, height, width) or(frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    alpha : np.ndarray, optional
        The alpha channel to use for blending, by default ``None``. Expected to be in range [0, 1], and of shape
        (T, X, Y) or (X, Y). If ``None``, the overlay array is used (if grayscale) or the alpha channel of the
        overlay video is used (if RGBA).
    fps : int, optional
        The framerate of the output video, by default 60
    skip_frames : int, optional
        Only export every ``skip_frames`` frames, by default 1
    cmap_base : str or matplotlib.colors.Colormap, optional
        The colormap to use for the base video, by default ``'gray'``
    cmap_overlay : str or matplotlib.colors.Colormap, optional
        The colormap to use for the overlay video, by default ``'Purples'``
    vmin_base : float, optional
        The minimum value for the base video, by default None
    vmax_base : float, optional
        The maximum value for the base video, by default None
    vmin_overlay : float, optional
        The minimum value for the overlay video, by default None
    vmax_overlay : float, optional
        The maximum value for the overlay video, by default None
    ffmpeg_encoder : str, optional
        The ffmpeg encoder to use, by default ``'libx264'``.
        See :func:`set_default_ffmpeg_encoder` and :func:`set_ffmpeg_defaults` for more information.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True
    """
    if isinstance(overlay, (str, os.PathLike)):
        filename, base, overlay = base, overlay, filename
        warnings.warn("WARNING: The order of arguments for export_video_with_overlay() has changed. "
                      "Please use export_video_with_overlay(filename, base, overlay, ...) instead of "
                      "export_video_with_overlay(base, overlay, filename, ...).",
                      DeprecationWarning)

    if ffmpeg_encoder is None:
        ffmpeg_encoder = DEFAULT_FFMPEG_ENCODER

    writer = FFmpegWriter(
        filename,
        inputdict={"-r": f"{fps}"},
        outputdict=_ffmpeg_defaults(ffmpeg_encoder),
    )

    for frame in tqdm(
        iter_alpha_blend_videos(
            base=base,
            overlay=overlay,
            alpha=alpha,
            skip_frames=skip_frames,
            cmap_base=cmap_base,
            cmap_overlay=cmap_overlay,
            vmin_base=vmin_base,
            vmax_base=vmax_base,
            vmin_overlay=vmin_overlay,
            vmax_overlay=vmax_overlay,
        ),
        desc="exporting video",
        total=len(base[::skip_frames]),
        disable=not progress_bar
    ):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        writer.writeFrame(frame)
    writer.close()
    print(f"video exported to {filename}")


def alpha_blend_videos(*args, **kwargs):
    """Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_.

    Wrapper around :func:`iter_alpha_blend_videos` that returns the blended video as a numpy array.

    Returns
    -------
    np.ndarray {frames, height, width, 4}
        The blended video as with rgba channels.
    """
    return np.stack(
        list(
            iter_alpha_blend_videos(
                *args,
                **kwargs,
            )
        )
    )
