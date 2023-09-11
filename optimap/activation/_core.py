import numpy as np

from ..utils import _print, print_bar

def compute_activation_map(video, threshold=0.5, fps=None):
    """
    Compute activation/isochrone map from video using simple thresholding.

    :param video: {t, x, y} ndarray
    :param threshold: {float} threshold
    :param fps: {float} frames per second (optional)
    :return: {x, y} ndarray
    """
    _print(f'computing activation map with {threshold=}')
    if video.ndim != 3:
        raise ValueError('video must be 3-dimensional')
    
    amap = np.argmax(video > threshold, axis=0)
    amap = amap.astype(np.float32)
    if fps is not None:
        amap = amap / fps
    
    # set all pixels to nan that never reach threshold
    amap[(amap == 0) & (video[0] < threshold)] = np.nan
    
    _print(f'minimum of activation_map: {np.nanmin(amap)}')
    _print(f'maximum of activation_map: {np.nanmax(amap)}')
    print_bar()
    return amap
