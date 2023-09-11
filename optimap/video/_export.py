from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm
import skvideo.io
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import Normalize

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


def _ffmpeg_defaults(encoder: str = "libx264"):
    if not encoder in FFMEG_DEFAULTS:
        raise ValueError(f"ffmpeg encoder {encoder} not supported")
    else:
        return FFMEG_DEFAULTS[encoder]


def set_default_ffmpeg_encoder(encoder: str):
    """
    Set the default ffmpeg encoder to use for exporting videos.

    Parameters
    ----------
    encoder : str
        The ffmpeg encoder to use. E.g. 'libx264' or 'h264_nvenc'."""
    global DEFAULT_FFMPEG_ENCODER
    if not encoder in FFMEG_DEFAULTS:
        raise ValueError(
            f"Encoder {encoder} unknown, add it first using set_ffmpeg_defaults()"
        )
    DEFAULT_FFMPEG_ENCODER = encoder


def get_default_ffmpeg_encoder():
    """
    Get the default ffmpeg encoder to use for exporting videos.

    Returns
    -------
    str
        The ffmpeg encoder to use. E.g. 'libx264' or 'h264_nvenc'."""
    return DEFAULT_FFMPEG_ENCODER


def set_ffmpeg_defaults(encoder: str, params: dict):
    """
    Set the default ffmpeg parameters to use for exporting videos.

    Parameters
    ----------
    encoder : str
        The ffmpeg encoder for which the parameters apply. E.g. 'libx264' or 'h264_nvenc'.
    params : dict
        The ffmpeg parameters to use. E.g. {'-c:v': 'libx264', '-crf': '15', '-preset': 'slower'}."""
    FFMEG_DEFAULTS[encoder] = params


def export_video(
    video: np.ndarray,
    filename: Union[str, Path],
    fps: int = 60,
    skip_frames: int = 1,
    cmap="gray",
    vmin=None,
    vmax=None,
    ffmpeg_encoder=None,
):
    """
    Export a video numpy array to a video file (e.g. ``.mp4``) using `ffmpeg <https://www.ffmpeg.org>`_.

    Parameters
    ----------
    video : np.ndarray
        The video to export. Should be of shape (frames, height, width, channels) or (frames, height, width).
        If the video is grayscale, the colormap will be applied. If it's an RGB video, its values should range from 0 to 1.
    filename : str or Path
        Video file path for writing.
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
    ffmpeg_encoder : str, optional
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder` and :func:`set_ffmpeg_defaults` for more information.
    """
    if ffmpeg_encoder is None:
        ffmpeg_encoder = DEFAULT_FFMPEG_ENCODER

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    writer = skvideo.io.FFmpegWriter(
        filename,
        inputdict={"-r": f"{fps}"},
        outputdict=_ffmpeg_defaults(ffmpeg_encoder),
    )
    for frame in tqdm(video[::skip_frames], desc="exporting video"):
        if frame.ndim == 2:
            frame = cmap(norm(frame))

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        writer.writeFrame(frame)
    writer.close()
    print(f"video exported to {filename}")


def smoothstep(x, vmin=0, vmax=1, N=2):
    """
    Smoothly clamps the input array to the range [0, 1] using the `smoothstep function <https://en.wikipedia.org/wiki/Smoothstep>`_. Useful for e.g. creating alpha channels.

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
    """
    Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_. Yields the blended video frame by frame.

    The base video is blended with the overlay video using an alpha channel. If no alpha channel is provided, the overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used (if it is RGBA). The alpha channel is expected to be in the range [0, 1], a value of 0 means that the base video is used, a value of 1 means that the overlay video is used.

    Parameters
    ----------
    base : np.ndarray
        The base video. Should be of shape (frames, height, width) or (frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    overlay : np.ndarray
        The overlay video. Should be of shape (frames, height, width) or(frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    alpha : np.ndarray, optional
        The alpha channel to use for blending, by default None
        If None, the overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used (if it is RGBA).
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
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder` and :func:`set_ffmpeg_defaults` for more information.

    Yields
    ------
    np.ndarray {height, width, 4}
        The blended video frame.
    """

    if alpha is None:
        if overlay.ndim == 4:
            alpha = overlay[..., 3, np.newaxis]
        elif overlay.ndim == 3:
            alpha = overlay[..., np.newaxis]
    else:
        if alpha.ndim == 2:
            alpha = alpha[np.newaxis, :, :, np.newaxis]
        elif alpha.ndim == 3:
            alpha = alpha[..., np.newaxis]

    if isinstance(cmap_base, str):
        cmap_base = plt.get_cmap(cmap_base)
    if isinstance(cmap_overlay, str):
        cmap_overlay = plt.get_cmap(cmap_overlay)

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

        frame = f_base * (1 - f_alpha) + f_overlay * f_alpha
        yield frame


def export_video_with_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    filename: Union[str, Path],
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
):
    """
    Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_ and exports it to a video file (e.g. ``.mp4``).

    The base video is blended with the overlay video using an alpha channel. If no alpha channel is provided, the overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used (if it is RGBA). The alpha channel is expected to be in the range [0, 1], a value of 0 means that the base video is used, a value of 1 means that the overlay video is used.

    Parameters
    ----------
    base : np.ndarray
        The base video. Should be of shape (frames, height, width) or (frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    overlay : np.ndarray
        The overlay video. Should be of shape (frames, height, width) or(frames, height, width, channels).
        Either uint8 or float32 (expected to be in the range [0, 1]).
    filename : str or Path
        Video file path for writing.
    alpha : np.ndarray, optional
        The alpha channel to use for blending, by default None
        If None, the overlay array is used as alpha channel (if it is grayscale) or the alpha channel of the overlay video is used (if it is RGBA).
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
        The ffmpeg encoder to use, by default ``'libx264'``. See :func:`set_default_ffmpeg_encoder` and :func:`set_ffmpeg_defaults` for more information.
    """
    if ffmpeg_encoder is None:
        ffmpeg_encoder = DEFAULT_FFMPEG_ENCODER

    writer = skvideo.io.FFmpegWriter(
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
    ):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        writer.writeFrame(frame)
    writer.close()
    print(f"video exported to {filename}")


def alpha_blend_videos(*args, **kwargs):
    """
    Blends two videos together using `alpha blending <https://en.wikipedia.org/wiki/Alpha_compositing>`_.

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
