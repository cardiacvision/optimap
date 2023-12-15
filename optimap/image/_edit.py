import cv2
import numpy as np


def resize(image, shape=None, scale=None, interpolation=cv2.INTER_CUBIC):
    """Resize image.

    Either shape or scale parameter must be specified.

    Parameters
    ----------
    image : image_like
        Image to resize. Either grayscale or RGB(A)
    shape : (new_x, new_y) tuple
        Shape of the desired image
    scale : float
        Scale factor to apply to image, e.g. 0.5 for half the size
    interpolation : int, optional
        Interpolation method to use, by default cv2.INTER_CUBIC. See OpenCV documentation for details.

    Returns
    -------
    2D np.ndarray
        Resized image.
    """
    if shape is None and scale is None:
        raise ValueError("Either shape or scale parameter must be specified.")
    if shape is not None and scale is not None:
        raise ValueError("Only one of shape and scale parameters can be specified.")
    return cv2.resize(image, dsize=shape, fx=scale, fy=scale, interpolation=interpolation)


def rotate_left(image, k=1):
    """Rotate image by 90° counter-clockwise (left).

    Parameters
    ----------
    image : {x, y} ndarray
        Image to rotate.
    k : int, optional
        Number of times to rotate by 90°, by default 1

    Returns
    -------
    {y, x} ndarray
        Rotated image.
    """
    return np.rot90(image, k=k)

def rotate_right(image, k=1):
    """Rotate image by 90° clockwise (right).

    Parameters
    ----------
    image : {x, y} ndarray
        Image to rotate.
    k : int, optional
        Number of times to rotate by 90°, by default 1

    Returns
    -------
    {y, x} ndarray
        Rotated image.
    """
    return np.rot90(image, k=-k)

def flip_up_down(image):
    """Flip image up-down.

    Parameters
    ----------
    image : {x, y} ndarray
        Image to flip.

    Returns
    -------
    {x, y} ndarray
        Flipped image.
    """
    return image[::-1]

def flip_left_right(image):
    """Flip image left-right.

    Parameters
    ----------
    image : {x, y} ndarray
        Image to flip.

    Returns
    -------
    {x, y} ndarray
        Flipped image.
    """
    return image[:, ::-1]

def crop(image, width):
    """Crop image by *width* pixels on each side.

    Parameters
    ----------
    image : {x, y} ndarray
        Image to crop.
    width : int
        Width of crop.

    Returns
    -------
    {t, x-2*width, y-2*width} ndarray
    """
    return image[width:-width, width:-width]

def pad(image, width, mode="constant" , **kwargs):
    """Pad image by *width* pixels on each side.

    Parameters
    ----------
    image : {x, y} ndarray
        Image to pad.
    width : int
        Width of padding.
    mode : str, optional
        Padding mode, by default 'constant'. See :py:func:`numpy.pad` for details and additional keyword arguments.
    **kwargs : dict, optional
        Additional keyword arguments passed to :py:func:`numpy.pad`.

    Returns
    -------
    {x+2*width, y+2*width} ndarray
    """
    if image.ndim == 2:
        padding = ((width,width), (width,width))
    elif image.ndim == 3:  # RGB(A) image
        padding = ((width,width), (width,width), (0,0))
    return np.pad(image, padding, mode=mode, **kwargs)
