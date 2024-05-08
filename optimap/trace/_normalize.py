import numpy as np

def normalize(array: np.ndarray, ymin=0, ymax=None, vmin=None, vmax=None, dtype=np.float32, clip=True):
    """Normalize an array (time-series or multiple-timeseries, 1D or 2D array) to a specified range and data type.

    By default, the input will be normalized to the interval [0, 1] with type np.float32 based on the minumum and maximum value of the input array.

    If parameters ``vmin`` or ``vmax`` are specified, the normalization is performed using these values and the resulting array will be clipped.

    The parameters ``ymin`` and ``ymax`` specify the minimum and maximum values of the resulting array, by default 0 and 1 if ``dtype`` is a floating point type, or the maximum value of the data type if ``dtype`` is an integer type.


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
        Normalized array/time-series (multiple).
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
        vmin = np.nanmin(array, axis=0)
    if vmax is None:
        vmax = np.nanmax(array, axis=0)

    eps = np.finfo(array.dtype).eps
    array = (array - vmin) / (vmax - vmin + eps) * (ymax - ymin) + ymin

    if do_clip:
        array = np.clip(array, ymin, ymax)

    if dtype == array.dtype:
        return array
    else:
        return array.astype(dtype)