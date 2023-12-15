import numpy as np

from ..image import resize as resize_image
from ..utils import _print, print_bar


def resize(video, shape=None, scale=None, interpolation="cubic"):
    """Resize video.

    Either shape or scale must be specified.

    Interpolation method is applied to each frame individually. See :func:`optimap.image.resize`
    for interpolation methods, "area" is recommended for downsampling as it gives moire-free results.

    Parameters
    ----------
    video : {t, x, y} np.ndarray
        Video to resize.
    shape : (new_x, new_y) tuple
        Spatial shape of resized video. Either this or scale must be specified.
    scale : float
        Scale factor to apply to video, e.g. 0.5 for half the size. Either this or shape must be specified.
    interpolation : str, optional
        Interpolation method to use, by default "cubic". See :func:`optimap.image.resize` for interpolation methods.

    Returns
    -------
    3D np.ndarray
        Resized video.
    """
    if shape is None and scale is None:
        raise ValueError("Either shape or scale parameter must be specified.")
    if shape is not None and scale is not None:
        raise ValueError("Only one of shape and scale parameters can be specified.")

    nt = video.shape[0]
    img0 = resize_image(video[0], shape=shape, scale=scale, interpolation=interpolation)
    video_new = np.zeros((nt,) + img0.shape, dtype=video.dtype)
    video_new[0] = img0
    for t in range(1, nt):
        video_new[t] = resize_image(video[t], shape=shape, scale=scale, interpolation=interpolation)
    return video_new


def rotate_left(video, k=1):
    """Rotate video by 90° counter-clockwise (left).

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to rotate.
    k : int, optional
        Number of times to rotate by 90°, by default 1

    Returns
    -------
    {t, y, x} ndarray
        Rotated video.
    """
    _print("rotating video 90 degree to the left (counter-clockwise)")
    video = np.rot90(video, k=k, axes=(1, 2))
    print_bar()
    return video

def rotate_right(video, k=1):
    """Rotate video by 90° clockwise (right).

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to rotate.
    k : int, optional
        Number of times to rotate by 90°, by default 1

    Returns
    -------
    {t, y, x} ndarray
        Rotated video.
    """
    _print("rotating video 90 degree to the right (clockwise)")
    video = np.rot90(video, k=-k, axes=(1, 2))
    print_bar()
    return video

def flip_up_down(video):
    """Flip Video up-down.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to flip.

    Returns
    -------
    {t, x, y} ndarray
        Flipped video.
    """
    return video[:, ::-1, :]

def flip_left_right(video):
    """Flip Video left-right.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to flip.

    Returns
    -------
    {t, x, y} ndarray
        Flipped video.
    """
    return video[:, :, ::-1]

def crop(video, width):
    """Crop Video by *width* pixels on each side.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to crop.
    width : int
        Width of crop.

    Returns
    -------
    {t, x-2*width, y-2*width} ndarray
    """
    _print("cropping array ...")
    video = video[:, width:-width, width:-width]
    print_bar()
    return video

def pad(video, width, mode="constant" , **kwargs):
    """Pad Video by *width* pixels on each side.

    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to pad.
    width : int
        Width of padding.
    mode : str, optional
        Padding mode, by default 'constant'. See :py:func:`numpy.pad` for details and additional keyword arguments.
    **kwargs : dict, optional
        Additional keyword arguments passed to :py:func:`numpy.pad`.

    Returns
    -------
    {t, x+2*width, y+2*width} ndarray
    """
    _print("padding array ...")
    video = np.pad(video, ((0,0), (width,width), (width,width)), mode=mode, **kwargs)
    print_bar()
    return video
