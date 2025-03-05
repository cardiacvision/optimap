import numpy as np

from ..image import collage as _collage_images
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
    video = video[:, width:-width, width:-width]
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

def collage(videos, ncols=6, padding=0, padding_value=0):
    """Creates a video collage from a list of videos with the same shape.

    Arranges the frames of the videos in a grid, similar to the :py:func:`optimap.image.collage` function. See also :func:`optimap.video.export_videos` to export a list of video as a collage to a mp4 file and :func`optimap.trace.collage_positions` to create a collage of positions.

    Parameters
    ----------
    videos : list of np.ndarray
        List of videos, where each video is a numpy array. All videos must have the same shape.
    ncols : int, optional
        Number of videos per row in the collage, by default 6.
    padding : int, optional
        Spacing between videos in pixels, by default 0.
    padding_value : float or np.ndarray, optional
        Value for the spacing (e.g., color as an RGB(A) array), by default 0.

    Returns
    -------
    np.ndarray
        Collage video as a 4D numpy array (T, H_collage, W_collage, C).

    Raises
    ------
    ValueError
        If the input videos do not have the same shape.

    Examples
    --------
    .. code-block:: python

        collage = om.video.collage([video1, video2, video3], ncols=2, padding=5)
        om.video.show_video(collage)

    """
    if not videos:
        raise ValueError("Input video list cannot be empty.")

    shape = videos[0].shape
    for video in videos:
        if video.shape != shape:
            raise ValueError("All videos must have the same shape.")

    collaged_frames = []
    for frame_idx in range(videos[0].shape[0]):
        frame_list = [video[frame_idx] for video in videos]
        collaged_frame = _collage_images(frame_list, ncols=ncols, padding=padding, padding_value=padding_value)
        collaged_frames.append(collaged_frame)

    return np.stack(collaged_frames, axis=0)
