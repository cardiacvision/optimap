from typing import Optional

import numpy as np
from scipy.signal import hilbert

from .. import _cpp
from ..utils import _print, print_bar
from ..video import play


def compute_phase(video, offset=-0.5):
    """Computes phase using Hilbert transformation, takes a normalized video or time-series as input.

    Parameters
    ----------
    video : {t, x, y} ndarray
        normalized video
    offset : float
        offset to add to video before computing phase, default is -0.5

    Returns
    -------
    {t, x, y} ndarray
        phase in range [-pi, pi]
    """
    _print("computing phase video (using hilbert transform and assuming normalized video with values [0,1]) ... ")
    analytic_signal = hilbert(video + offset, axis=0)
    phase = np.angle(analytic_signal).astype(np.float32)
    _print("minimum of phase: " + str(np.nanmin(phase)))
    _print("maximum of phase: " + str(np.nanmax(phase)))
    print_bar()
    return phase

def phasefilter_angle_threshold(phase: np.ndarray, wx: int, wy: int, wt: int, tr_angle: float,
                                mask: Optional[np.ndarray] = None):
    """Remove outliers in a phase video by comparing them against their neighbors in space and time.

    Parameters
    ----------
    phase : {t, x, y} ndarray
        phase video
    wx : int
        window size in x-direction
    wy : int
        window size in y-direction
    wt : int
        window size in t-direction
    tr_angle : float
        threshold angle in radians
    mask : {x, y} ndarray, optional
        only consider pixels where mask is True

    Returns
    -------
    {t, x, y} ndarray
        phase video with outliers removed
    """
    if mask is None:
        return _cpp.phasefilter_angle_threshold(phase, wx, wy, wt, tr_angle)
    else:
        return _cpp.phasefilter_angle_threshold(phase, wx, wy, wt, tr_angle, mask)


def phasefilter_disc(phase: np.ndarray, wx: int, wy: int, wt: int, mask: Optional[np.ndarray] = None):
    """TODO: Docstring for phasefilter_disc."""
    if mask is None:
        return _cpp.phasefilter_disc(phase, wx, wy, wt)
    else:
        return _cpp.phasefilter_disc(phase, wx, wy, wt, mask)


def phasefilter_fillsmooth(phase: np.ndarray, wx: int, wy: int, wt: int, threshold: float,
                           mask: Optional[np.ndarray] = None):
    """Fills holes in phase video by smoothing over space and time.

    Parameters
    ----------
    phase : {t, x, y} ndarray
        phase video
    wx : int
        window size in x-direction
    wy : int
        window size in y-direction
    wt : int
        window size in t-direction
    threshold : float
        threshold for filling holes TODO??
    mask : {x, y} ndarray, optional
        only consider pixels where mask is True

    Returns
    -------
    {t, x, y} ndarray
        phase video with holes filled
    """
    if mask is None:
        return _cpp.phasefilter_fillsmooth(phase, wx, wy, wt, threshold)
    else:
        return _cpp.phasefilter_fillsmooth(phase, wx, wy, wt, threshold, mask)


def play_phase(video, skip_frame = 1, title="phase"):
    """Play a phase video using matplotlib."""
    return play(video, skip_frame, title, vmin=-np.pi, vmax=np.pi, cmap="hsv")
