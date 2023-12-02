import warnings

import numpy as np
from scipy import ndimage

from .. import _cpp
from ..utils import _print


def normalize(video: np.ndarray, ymin=0, ymax=1, vmin=None, vmax=None, clip=True):
    """Normalizes video to interval [``ymin``, ``ymax``]. If ``vmin`` or ``vmax`` are specified,
    the normalization is done using these values and the resulting video will be clipped.

    Parameters
    ----------
    video : {t, x, y} ndarray
        The input video to be normalized.
    ymin : float, optional
        Minimum value of the resulting video, by default 0
    ymax : float, optional
        Maximum value of the resulting video, by default 1
    vmin : float, optional
        Minimum value of the input video, by default None
        If None, the minimum value of the input video is calculated.
    vmax : float, optional
        Maximum value of the input video, by default None
        If None, the maximum value of the input video is calculated.
    clip : bool, optional
        If True, the resulting video will be clipped to [``ymin``, ``ymax``], by default True
        Only applies if ``vmin`` or ``vmax`` are specified.

    Returns
    -------
    {t, x, y} ndarray
        Normalized video.
    """
    _print(f"normalizing video to interval [{ymin}, {ymax}] ...")
    video = video.astype("float32")
    do_clip = clip and (vmin is not None or vmax is not None)
    if vmin is None:
        vmin = np.nanmin(video)
    if vmax is None:
        vmax = np.nanmax(video)

    eps = np.finfo(np.float32).eps
    video = (video - vmin) / (vmax - vmin + eps) * (ymax - ymin) + ymin

    if do_clip:
        video = np.clip(video, ymin, ymax)
    return video


def normalize_pixelwise(video: np.ndarray, ymin=0, ymax=1):
    """Normalizes video pixel-wise to interval [ymin, ymax].

    Parameters
    ----------
    video : {t, x, y} ndarray
        The input video to be normalized.
    ymin : float, optional
        Minimum value, by default 0
    ymax : float, optional
        Maximum value, by default 1

    Returns
    -------
    {t, x, y} ndarray
        Normalized video.
    """
    _print(f"normalizing video pixel-wise to interval [{ymin}, {ymax}] ...")
    video = video.astype("float32")
    with warnings.catch_warnings():  # ignore "All-NaN slice encountered" warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_ = np.nanmin(video, axis=0)
        max_ = np.nanmax(video, axis=0)
    eps = np.finfo(np.float32).eps
    return (video - min_) / (max_ - min_ + eps) * (ymax - ymin) + ymin


def normalize_pixelwise_slidingwindow(video: np.ndarray, window_size: int, ymin=0, ymax=1):
    """Normalizes video pixel-wise using a temporal sliding window.

    For each frame ``t`` in the video, this function normalizes its pixels based on the pixel values within a window
    spanning from ``[t - window_size//2, t + window_size//2]``. The normalization maps pixel values to the interval
    ``[ymin, ymax]``.

    .. note::
        If `window_size` if even, the window size will be increased by 1 to make it odd.
        This is to ensure that the window is symmetric around the current frame ``t``.

        The window shrinks at the beginning and end of the video, where there are not enough frames to fill the window.

    Parameters
    ----------
    video : {t, x, y} ndarray
        The input video to be normalized.
    window_size : int
        The size of the sliding window.
    ymin : float, optional
        The minimum value of the normalization interval. Default is 0.
    ymax : float, optional
        The maximum value of the normalization interval. Default is 1.

    Returns
    -------
    {t, x, y} ndarray
        Normalized video.
    """
    _print(f"normalizing video pixel-wise using sliding window of size {2*(window_size//2)+1} ...")
    if video.ndim != 3:
        msg = "ERROR: video has to be 3 dimensional"
        raise ValueError(msg)
    return _cpp.normalize_pixelwise_slidingwindow(video, window_size // 2, ymin, ymax)


def smooth_spatiotemporal(video: np.ndarray, sigma_temporal, sigma_spatial):
    """Smooth video using a Gaussian filter in space and time.

    Parameters
    ----------
    video : {t, x, y} ndarray
        video to smooth
    sigma_temporal : float
        Standard deviation for Gaussian kernel in time.
    sigma_spatial : float
        Standard deviation for Gaussian kernel in space.

    Returns
    -------
    {t, x, y} ndarray
        Filtered video.
    """
    if video.ndim != 3:
        msg = "ERROR: video has to be 3 dimensional"
        raise ValueError(msg)
    return ndimage.gaussian_filter(
        video, sigma=(sigma_temporal, sigma_spatial, sigma_spatial), order=0
    )


def temporal_difference(video: np.ndarray, n: int):
    """Temporal difference filter. Computes difference between frames using an offset of `n` frames.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to filter.
    n : int
        Offset

    Returns
    -------
    {t, x, y} ndarray
        Filtered video.
    """
    if video.dtype != np.float32:
        video = video.astype(np.float32)
    diff = video[n:] - video[:-n]
    z = np.zeros((n,) + video.shape[1:], dtype=np.float32)
    return np.vstack((z, diff))


def evolve_jitter_filter(video, framerate=500.0, threshold=0.004):
    """Jitter removal filter for Photometrics Evolve 128 camera.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to filter.
    framerate : float
        Framerate in Hz.
    threshold : float
        Threshold.

    Returns
    -------
    {t, x, y} ndarray
        Filtered video.
    """
    video = video[..., None].copy()
    factor = 1 + 14e-6 * framerate

    ts = video.mean((1, 2, 3))
    tsd = np.diff(ts) / np.median(ts)

    for ii in range(len(tsd) - 1):
        if (tsd[ii] < -threshold) & (tsd[ii + 1] > threshold):
            np.multiply(video[ii + 1], factor, out=video[ii + 1], casting="unsafe")
    return np.squeeze(video)
