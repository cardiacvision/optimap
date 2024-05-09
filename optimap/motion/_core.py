import numpy as np

from .. import _cpp
from ..video import smooth_spatiotemporal
from ._flowestimator import FlowEstimator
from ._warping import warp_video


def contrast_enhancement(video_or_img: np.ndarray, kernel_size: int, mask: np.ndarray = None):
    """Amplifies local contrast to maximum to remove fluorescence signal for motion estimation.
    See :cite:t:`Christoph2018a` for details.

    Parameters
    ----------
    video_or_img : np.ndarray
        {t, x, y} or {x, y} ndarray
    kernel_size : int
        Kernel size for local contrast enhancement (must be odd)
    mask : np.ndarray, optional
        valid values mask of shape {x, y} or {t, x, y}, by default None

    Returns
    -------
    np.ndarray
        {t, x, y} or {x, y} ndarray of dtype np.float32
    """
    if video_or_img.ndim == 2:
        if mask is None:
            return _cpp.contrast_enhancement_img(np.ascontiguousarray(video_or_img), kernel_size)
        else:
            return _cpp.contrast_enhancement_img(np.ascontiguousarray(video_or_img), kernel_size, np.ascontiguousarray(mask))
    else:
        if mask is None:
            return _cpp.contrast_enhancement_video(np.ascontiguousarray(video_or_img), kernel_size)
        else:
            if mask.ndim == 2:
                mask = mask[None, ...].repeat(video_or_img.shape[0], axis=0)
            return _cpp.contrast_enhancement_video(np.ascontiguousarray(video_or_img), kernel_size, np.ascontiguousarray(mask))


def smooth_displacements(
    displacements,
    wx: int,
    wy: int,
    wt: int,
    mask: np.ndarray = np.array([[0.0]], dtype=np.float32),
):
    """Smooths optical flow fields in space and time using a Gaussian kernel.

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
    """Calculate optical flow between every frame of a video and a reference frame. Wrapper around
    :py:class:`FlowEstimator` for convenience. See :py:meth:`FlowEstimator.estimate` for details.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y})
    ref_frame : int, optional
        Index of reference frame to estimate optical flow to, by default 0
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use (default: ``'farneback'`` if GPU is available, ``'farneback_cpu'`` otherwise),
        by default None

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
    """Calculate optical flow between every frame of a video and a reference frame. Wrapper around
    :py:class:`FlowEstimator` for convenience. See :py:meth:`FlowEstimator.estimate_reverse` for details.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y})
    ref_frame : int, optional
        Index of reference frame to estimate optical flow to, by default 0
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use, by default ``None`` which means ``'farneback'`` if a CUDA GPU is
        available, or ``'farneback_cpu'`` otherwise

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
    """Typical motion compensation pipeline for a optical mapping video.

    First, the video is smoothed in space and time using a Gaussian kernel. Then, local contrast is enhanced to
    remove fluorescence signal. Finally, optical flow is estimated between every frame and a reference frame,
    and the video is warped to the reference frame using the estimated optical flow.

    See :py:func:`motion.contrast_enhancement` and :py:func:`motion.estimate_displacements` for further details.

    Parameters
    ----------
    video : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y}). Can be any dtype because
        contrast enhancement will convert it to float32.
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
        Tuple of (wx, wy, wt) for Gaussian smoothing kernel in space and time, by default None.
        If None, no smoothing is applied. See :py:func:`smooth_displacements` for details.
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use, by default ``None`` which means ``'farneback'`` if a CUDA GPU is
        available, or ``'farneback_cpu'`` otherwise

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
    return warp_video(original_video, flows)


def reverse_motion_compensate(
    video_tracking,
    video_warping,
    contrast_kernel=7,
    ref_frame=0,
    presmooth_temporal=0.8,
    presmooth_spatial=0.8,
    postsmooth=None,
    method=None,
):
    """Typical motion tracking pipeline to transform a video back into motion.
    E.g. we first motion compensated a recording and extracted the fluorescence wave dynamics.
    We now want to transform the processed, motion-less, video back into motion and e.g.
    overlay it on-top the original video with :py:func:`video.play_with_overlay`.

    See :py:func:`motion_compensate` and :py:func:`estimate_reverse_displacements` for explanation
    of parameters and further details.

    Parameters
    ----------
    video_tracking : np.ndarray
        Video to estimate optical flow for (list of images or 3D array {t, x, y}). Can be any dtype.
    video_warping : np.ndarray
        Video to be warped based on the motion of the `video_tracking` data.
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
        Tuple of (wx, wy, wt) for Gaussian smoothing kernel in space and time, by default None.
        If None, no smoothing is applied. See :py:func:`smooth_displacements` for details.
    show_progress : bool, optional
        Show progress bar, by default None
    method : str, optional
        Optical flow method to use, by default ``None`` which means ``'farneback'`` if a CUDA GPU is
        available, or ``'farneback_cpu'`` otherwise

    Returns
    -------
    np.ndarray

    """
    if presmooth_temporal > 0 or presmooth_spatial > 0:
        video_tracking = video_tracking.astype(np.float32)
        video_tracking = smooth_spatiotemporal(video_tracking, presmooth_temporal, presmooth_spatial)
    if contrast_kernel != 0:
        video_tracking = contrast_enhancement(video_tracking, contrast_kernel)
    flows = estimate_reverse_displacements(video_tracking, ref_frame, method=method)
    if postsmooth is not None:
        flows = smooth_displacements(flows, *postsmooth)
    return warp_video(video_warping, flows)
