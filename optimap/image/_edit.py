import cv2
import numpy as np

MAP_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

def normalize(array: np.ndarray, ymin=0, ymax=None, vmin=None, vmax=None, dtype=np.float32, clip=True):
    """Normalize an array (video, image, ...) to a specified range and data type.

    By default, the input will be normalized to the interval [0, 1] with type np.float32 based on the minumum and maximum value of the input array.

    If parameters ``vmin`` or ``vmax`` are specified, the normalization is performed using these values and the resulting array will be clipped.

    The parameters ``ymin`` and ``ymax`` specify the minimum and maximum values of the resulting array, by default 0 and 1 if ``dtype`` is a floating point type, or the maximum value of the data type if ``dtype`` is an integer type.

    Examples
    --------
    .. code-block:: python

        import optimap as om
        import numpy as np

        filepath = om.download_example_data("Sinus_Rabbit_1.npy")
        video = om.load_video(filepath)

        # normalize video to interval [0, 1] using the minimum and maximum values of the video
        video_normalized = om.video.normalize(video)

        # normalize video to interval [0, 255] by converting the video to uint8
        video_normalized_uint8 = om.video.normalize(video, ymin=0, ymax=255, dtype=np.uint8)

    Parameters
    ----------
    array : ndarray
        The input array to be normalized.
    ymin : float, optional
        Minimum value of the resulting video, by default 0
    ymax : float, optional
        Maximum value of the resulting video, by default 1 for floating point arrays, or the maximum value of the data type for integer arrays.
    vmin : float, optional
        Minimum value of the input video, by default None
        If None, the minimum value of the input video is calculated.
    vmax : float, optional
        Maximum value of the input video, by default None
        If None, the maximum value of the input video is calculated.
    dtype : type, optional
        Data type of the resulting array, by default np.float32
    clip : bool, optional
        If True, the resulting video will be clipped to [``ymin``, ``ymax``], by default True
        Only applies if ``vmin`` or ``vmax`` are specified.

    Returns
    -------
    ndarray
        Normalized array/video/image.
    """
    do_clip = clip and (vmin is not None or vmax is not None)
    dtype = np.dtype(dtype)

    if ymax is None:
        if dtype.kind in ["u", "i"]:
            ymax = np.iinfo(dtype).max
        else:
            ymax = 1.0

    if not (np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.complexfloating)):
        array = array.astype(np.float32)

    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)

    eps = np.finfo(array.dtype).eps
    array = (array - vmin) / (vmax - vmin + eps) * (ymax - ymin) + ymin

    if do_clip:
        array = np.clip(array, ymin, ymax)

    if dtype == array.dtype:
        return array
    else:
        return array.astype(dtype)


def resize(image, shape=None, scale=None, interpolation="cubic"):
    """Resize image.

    Either shape or scale parameter must be specified.

    Interpolation methods:
        - nearest: nearest neighbor interpolation
        - linear: bilinear interpolation
        - cubic: bicubic interpolation
        - area: resampling using pixel area relation. It may be a preferred method for image decimation, as it
          gives moire'-free results. But when the image is zoomed, it is similar to the "nearest" method.
        - lanczos: a Lanczos interpolation over 8x8 neighborhood

    Parameters
    ----------
    image : image_like
        Image to resize. Either grayscale or RGB(A)
    shape : (new_x, new_y) tuple
        Shape of the desired image
    scale : float
        Scale factor to apply to image, e.g. 0.5 for half the size
    interpolation : str, optional
        Interpolation method to use, by default "cubic". See above for available methods.

    Returns
    -------
    2D np.ndarray
        Resized image.
    """
    if shape is None and scale is None:
        raise ValueError("Either shape or scale parameter must be specified.")
    if shape is not None and scale is not None:
        raise ValueError("Only one of shape and scale parameters can be specified.")
    if interpolation not in MAP_INTERPOLATION:
        raise ValueError(f"Unknown interpolation method: {interpolation}. Must be one of {MAP_INTERPOLATION.keys()}.")

    return cv2.resize(image, dsize=shape, fx=scale, fy=scale, interpolation=MAP_INTERPOLATION[interpolation])


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
    {x-2*width, y-2*width} ndarray
    """
    if width == 0:
        return image
    else:
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
