import numpy as np

from ..utils import _print, print_bar

def compute_activation_map(video, threshold=0.5, inverted=False, fps=None, set_nan_for_inactive=True):
    """
    Computes an activation map (or isochrone map) from a given video based on pixel intensity thresholding.
    
    For each pixel in the video, the function determines the time (or frame index) at which the pixel's intensity first surpasses (or falls below, if inverted is set to True) the specified threshold.
    
    Parameters
    ----------
    video : ndarray
        A 3D array representing the video, with dimensions {t (time or frames), x (width), y (height)}.
    threshold : float, optional
        Intensity threshold at which a pixel is considered activated. Defaults to 0.5.
    inverted : bool, optional
        If True, the function will compute the time/frame when pixel intensity falls below the threshold, 
        rather than surpassing it. Defaults to False.
    fps : float, optional
        If provided, the resulting activation map will represent times in seconds based on this frame rate,
        otherwise, it will be in frames.
    set_nan_for_inactive : bool, optional
        If True, pixels that never reach the activation threshold will be set to NaN. Defaults to True.
    
    Returns
    -------
    activation_map : ndarray
        2D image
    """
    _print(f'computing activation map with {threshold=}')
    if video.ndim != 3:
        raise ValueError('video must be 3-dimensional')

    if inverted:
        amap = np.nanargmax(video < threshold, axis=0)
    else:
        amap = np.nanargmax(video > threshold, axis=0)
    
    amap = amap.astype(np.float32)
    if fps is not None:
        amap = amap / fps

    if set_nan_for_inactive:
        # set all pixels to NaN that never reach threshold
        if inverted:
            never_activated = (amap == 0) & (video[0] > threshold)
        else:
            never_activated = (amap == 0) & (video[0] < threshold)
        amap[never_activated] = np.nan

    # set masked pixels to NaN
    amap[np.isnan(video[0])] = np.nan
    
    _print(f'minimum of activation_map: {np.nanmin(amap)}')
    _print(f'maximum of activation_map: {np.nanmax(amap)}')
    print_bar()
    return amap
