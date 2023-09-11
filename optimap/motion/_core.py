import numpy as np

from .. import _cpp
from ..video import smooth_spatiotemporal
from ._warping import warp_video
from ._flowestimator import FlowEstimator


def contrast_enhancement(video_or_img: np.ndarray, kernel_size: int):
    """
    Amplifies local contrast to maximum to remove fluorescence signal for motion estimation. See :cite:t:`Christoph2018a` for details.

    :param video_or_img: {t, x, y} or {x, y} ndarray
    :param kernel_size: int kernel size for local contrast enhancement (must be odd)
    :return: {t, x, y} or {x, y} ndarray
    """
    if video_or_img.ndim == 2:
        return _cpp.contrast_enhancement_img(np.ascontiguousarray(video_or_img), kernel_size)
    else:
        return _cpp.contrast_enhancement_video(np.ascontiguousarray(video_or_img), kernel_size)


def smooth_displacements(
    displacements,
    wx: int,
    wy: int,
    wt: int,
    mask: np.ndarray = np.array([[0.0]], dtype=np.float32),
):
    """
    Smooths optical flow fields in space and time using a Gaussian kernel.

    Parameters
    ----------
    displacements : np.ndarray
        {t, x, y, 2} optical flow array
    wx : int
        kernel size in x-direction
    wy : int
        kernel size in y-direction
    wt : int
        kernel size in t-direction
    mask : np.ndarray, optional
        {x, y} mask to apply to flow field before smoothing, by default none

    Returns
    -------
    np.ndarray
        {t, x, y, 2} smoothed optical flow array
    """
    return _cpp.flowfilter_smooth_spatiotemporal(displacements, wx, wy, wt, mask)


def estimate_displacements(video, ref_frame=0, show_progress=True, method=None):
    """Calculate optical flow between every frame of a video and a reference frame. Wrapper around :py:class:`FlowEstimator` for convenience. See :py:meth:`FlowEstimator.estimate` for details.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y})
    ref_frame : int, optional
        Index of reference frame to estimate optical flow to, by default 0
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use (default: 'farneback' if GPU is available, 'farneback_cpu' otherwise), by default None

    Returns
    -------
    np.ndarray
        Optical flow array of shape {t, x, y, 2}
    """
    estimator = FlowEstimator()
    return estimator.estimate(
        video, video[ref_frame], show_progress=show_progress, method=method
    )


def estimate_reverse_displacements(video, ref_frame=0, show_progress=True, method=None):
    """Calculate optical flow between every frame of a video and a reference frame. Wrapper around :py:class:`FlowEstimator` for convenience. See :py:meth:`FlowEstimator.estimate_reverse` for details.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y})
    ref_frame : int, optional
        Index of reference frame to estimate optical flow to, by default 0
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use (default: 'farneback' if GPU is available, 'farneback_cpu' otherwise), by default None

    Returns
    -------
    np.ndarray
        Optical flow array of shape {t, x, y, 2}
    """
    estimator = FlowEstimator()
    return estimator.estimate_reverse(
        video, video[ref_frame], show_progress=show_progress, method=method
    )


def motion_compensate(
    video,
    contrast_kernel=7,
    ref_frame=0,
    presmooth_temporal=0.8,
    presmooth_spatial=0.8,
    postsmooth=None,
    method=None,
):
    """Typical motion compensation pipeline for a optical mapping video. See :py:func:`contrast_enhancement` and :py:func:`estimate_flows` for details.

    First, the video is smoothed in space and time using a Gaussian kernel. Then, local contrast is enhanced to remove fluorescence signal. Finally, optical flow is estimated between every frame and a reference frame, and the video is warped to the reference frame using the estimated optical flow.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y}). Can be any dtype because contrast enhancement will convert to float32.
    contrast_kernel : int, optional
        Kernel size for local contrast enhancement (must be odd), by default 7
        See :py:func:`contrast_enhancement` for details.
    ref_frame : int, optional
        Index of reference frame to estimate optical flow to, by default 0
    presmooth_temporal : float, optional
        Standard deviation of smoothing Gaussian kernel in time, by default 0.8
        See :py:func:`optimap.video.smooth_spatiotemporal` for details.
    presmooth_spatial : float, optional
        Standard deviation of smoothing Gaussian kernel in space, by default 0.8
        See :py:func:`optimap.video.smooth_spatiotemporal` for details.
    postsmooth : tuple, optional
        Tuple of (wx, wy, wt) for Gaussian smoothing kernel in space and time, by default None. If None, no smoothing is applied. See :py:func:`smooth_displacements` for details.
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use (default: 'farneback' if GPU is available, 'farneback_cpu' otherwise), by default None

    Returns
    -------
    np.ndarray
        Motion-compensated video of shape {t, x, y}
    """
    original_video = video

    if presmooth_temporal > 0 or presmooth_spatial > 0:
        video = video.astype(np.float32)
        video = smooth_spatiotemporal(video, presmooth_temporal, presmooth_spatial)
    if contrast_kernel != 0:
        video = contrast_enhancement(video, contrast_kernel)
    flows = estimate_displacements(video, ref_frame, method=method)
    if postsmooth is not None:
        flows = smooth_displacements(flows, *postsmooth)
    video_warped = warp_video(original_video, flows)
    return video_warped
