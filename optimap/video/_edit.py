import numpy as np
import cv2

from ..utils import _print, print_bar

def resize(video, newx, newy, interpolation=cv2.INTER_CUBIC):
    """
    Resize Video.

    Parameters
    ----------
    video : {t, x, y} np.ndarray
        Video to resize.
    newx : int
        New size in x-direction.
    newy : int
        New size in y-direction.
    interpolation : int, optional
        Interpolation method to use, by default cv2.INTER_CUBIC
    
    Returns
    -------
    {t, newx, newy} np.ndarray
        Resized video.
    """
    _print(f'resizing video to [{newx}, {newy}]')
    nt = video.shape[0]
    video_new = np.zeros((nt,newx,newy), dtype=video.dtype)
    for t in range(nt):
        video_new[t] = cv2.resize(video[t], dsize=(newx, newy), interpolation=interpolation)
    print_bar()
    return video_new


def rotate_left(video, k=1):
    """
    Rotate video by 90° counter-clockwise (left).
    
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
    _print('rotating video 90 degree to the left (counter-clockwise)')
    video = np.rot90(video, k=k, axes=(1, 2))
    print_bar()
    return video

def rotate_right(video, k=1):
    """
    Rotate video by 90° clockwise (right).

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
    _print('rotating video 90 degree to the right (clockwise)')
    video = np.rot90(video, k=-k, axes=(1, 2))
    print_bar()
    return video

def flip_up_down(video):
    """
    Flip Video up-down.
    
    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to flip.
    
    Returns
    -------
    {t, x, y} ndarray
        Flipped video.
    """
    video = video[:, ::-1, :]
    return video

def flip_left_right(video):
    """
    Flip Video left-right.
    
    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to flip.
    
    Returns
    -------
    {t, x, y} ndarray
        Flipped video.
    """
    video = video[:, :, ::-1]
    return video

def crop(video, width):
    """
    Crop Video by *width* pixels on each side.
    
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
    _print('cropping array ...')
    video = video[:, width:-width, width:-width]
    print_bar()
    return video

def pad(video, width, mode='constant' , **kwargs):
    """
    Pad Video by *width* pixels on each side.
    
    Parameters
    ----------
    video : {t, x, y} ndarray
        Video to pad.
    width : int
        Width of padding.
    mode : str, optional
        Padding mode, by default 'constant'. See :py:func:`numpy.pad` for details and additional keyword arguments.
    
    Returns
    -------
    {t, x+2*width, y+2*width} ndarray
    """
    _print('padding array ...')
    video = np.pad(video, ((0,0), (width,width), (width,width)), mode=mode, **kwargs)
    print_bar()
    return video