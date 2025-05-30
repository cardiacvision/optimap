import warnings

import numpy as np
from scipy import ndimage

from .. import _cpp
from ..utils import _print


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
    if video.ndim != 3:
        msg = "ERROR: video has to be 3 dimensional"
        raise ValueError(msg)
    video = np.ascontiguousarray(video)
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


def temporal_difference(array: np.ndarray, n: int, fill_value: float = 0, center: bool = False):
    # ruff: noqa: D301
    """Temporal difference filter using an offset of `n` frames.

    Computes temporal intensity changes in the videos or traces between frames at time :math:`t` and :math:`t - \\Delta t`:

    .. math::
        \\text{signal}_{\\text{diff}}(t) = \\text{signal}(t) - \\text{signal}(t - \\Delta t)

    where :math:`\\Delta t` is the time difference between frames (usually set to :math:`\\Delta t = 1-5` frames).

    The resulting signal is padded with `fill_value` to keep the original shape. If `center` is set to `True`, the padding is centered around the signal, otherwise it is added to the beginning.

    Parameters
    ----------
    array : ndarray
        Video or signal to filter. First axis is assumed to be time.
    n : int
        Offset in frames.
    fill_value : float, optional
        Value to fill the padded frames with, by default 0
    center : bool, optional
        If True, the padding is centered around the signal, by default False


    Returns
    -------
    ndarray
        Filtered video/signal.
    """
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating)):
        array = array.astype(np.float32)

    diff = array[n:] - array[:-n]
    if center:
        padding = ((n - n//2, n//2),)
    else:
        padding = ((n, 0),)
    diff = np.pad(diff, padding + ((0, 0),) * (diff.ndim - 1), constant_values=fill_value)
    return diff


def mean_filter(video: np.ndarray, size_temporal: int = 1, size_spatial: int = 1, mode: str = "nearest"):
    """Mean filter a video to smooth it (also known as uniform filter).
    
    The mean filter is optionally applied in space and/or time to smooth the video. Uses :func:`scipy.ndimage.uniform_filter`.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to filter.
    size_temporal : int
        Size of the temporal kernel.
    size_spatial : int
        Size of the spatial kernel.
    mode : str
        The mode parameter determines how the input array is extended when the filter overlaps a border.
        See :func:`scipy.ndimage.uniform_filter` for details. Default is 'nearest'.

    Returns
    -------
    {t, x, y} ndarray
        Filtered video.
    """
    if video.ndim != 3:
        msg = "ERROR: video has to be 3 dimensional"
        raise ValueError(msg)
    nans = np.isnan(video)
    if np.any(nans):
        # ndimage.uniform_filter doesn't handle NaNs, so we need to do it
        # TODO: Use ndimage.vectorized_filter when it is released (https://github.com/scipy/scipy/pull/22575)
        # or `nan_policy` when https://github.com/scipy/scipy/pull/17393 if it is ever merged
        video = video.copy()
        for i in range(video.shape[0]):
            video[i][np.isnan(video[i])] = np.nanmean(video[i])
    filtered = ndimage.uniform_filter(video, size=(size_temporal, size_spatial, size_spatial), mode=mode)
    filtered[nans] = np.nan
    return filtered


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
