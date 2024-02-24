import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..image import disc_mask

TRACE_TYPE = "rect"


def set_default_trace_window(window):
    """Set the default trace window type.

    Parameters
    ----------
    window : str
        Type of trace, one of 'rect', 'disc', 'pixel'
        See :py:func:`extract_trace` for more information.
    """
    if window not in ["rect", "disc", "pixel"]:
        msg = f"Unknown trace type {window}"
        raise ValueError(msg)

    global TRACE_TYPE
    TRACE_TYPE = window


def get_default_trace_window():
    """Get the default trace window.

    Returns
    -------
    str
        Type of trace, one of 'rect', 'disc', 'pixel'
    """
    return TRACE_TYPE


def extract_traces(video, coords, size=5, show=False, window=None, **kwargs):
    """Extract trace or traces from a video at specified coordinates.

    Multiple coordinates can be provided, and the averaging method for trace extraction can be selected.

    .. warning:: Coordinates are given as (y, x) tuples, not (x, y) tuples to be consistent with matplotlib convention (origin of image in the top left corner). E.g. the coordinate (5, 10) is equivalent to ``img[10, 5]`` in numpy.

    Different averaging window types of traces:

    - 'rect' - rectangle around the coordinates of size ``(size, size)``. This is the default option, see :py:func:`set_default_trace_window` to change the default. Note that if ``size`` is even the rectangle is shifted by one pixel to the top left, as there is no center pixel.
    - 'disc' - disc around the coordinates with diameter ``size``
    - 'pixel' - only the pixel at the coordinates (equivalent to ``size=1``)

    Parameters
    ----------
    video : 3D array
        Video to extract traces from
    coords : tuple or list of tuples
        Coordinates of the traces
    size : int, optional
        Size parameter for selection mask, by default 5px.
        If 0, only the pixel trace is returned.
    show : bool, optional
        Whether to show the traces using :py:func:`show_traces`, by default False
    window : str, optional
        Type of trace, by default 'rect'
        'rect' - rectangle around the coordinates
        'disc' - disc around the coordinates
        'pixel' - only the pixel at the coordinates (equivalent to size=0)
    **kwargs: dict
        Extra arguments for :py:func:`show_traces`

    Returns
    -------
    ndarray
        2D array containing the extracted traces.
    """  # noqa: E501
    single_coord = False
    if len(coords) == 0:
        return
    elif len(coords) == 2 and isinstance(coords[0], int):
        coords = [coords]
        single_coord = True

    if window is None:
        window = TRACE_TYPE

    for (y, x) in coords:
        if x < 0 or x >= video.shape[1] or y < 0 or y >= video.shape[2]:
            msg = f"Coordinates ({x}, {y}) out of bounds"
            raise ValueError(msg)

    def rect_mask(video, x, y):
        xs, xe = max(0, x - size // 2), min(video.shape[1], x + size // 2 + size % 2)
        ys, ye = max(0, y - size // 2), min(video.shape[2], y + size // 2 + size % 2)

        return video[:, xs:xe, ys:ye]

    if size == 0 or size == 1 or window == "pixel":
        traces = [video[:, x, y] for (y, x) in coords]
    elif window == "rect":
        traces = [rect_mask(video, x, y).mean(axis=(1, 2)) for (y, x) in coords]
    elif window == "disc":
        traces = [
            video[:, disc_mask(video.shape[1:], (x, y), size / 2)].mean(axis=(1,))
            for (y, x) in coords
        ]
    else:
        msg = f"Unknown trace window type '{window}'"
        raise ValueError(msg)
    traces = np.array(traces).T

    if single_coord:
        traces = traces[..., 0]

    if show or "ax" in kwargs:
        show_traces(traces, **kwargs)
    return traces


def show_positions(positions, image=None, size=None, color=None, cmap="gray", vmin=None, vmax=None, ax=None, **kwargs):
    """Overlay positions on an image.

    Parameters
    ----------
    positions : list of tuples
        List of positions to overlay
    image : 2D array
        Image to overlay positions on, optional
    size : float or array-like, shape (n, ), optional
        Size of the points, see `s` parameter in :py:func:`matplotlib.pyplot.scatter` for more information
    color : str or list of str, optional
        Color of the points, by default None
    cmap : str, optional
        Colormap to use for image, by default 'gray'
    vmin : float, optional
        Minimum value for the colormap, by default None
    vmax : float, optional
        Maximum value for the colormap, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    kwargs : dict, optional
        Additional arguments to pass to :py:func:`matplotlib.pyplot.scatter`

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False
    
    if isinstance(image, np.ndarray) and image.ndim == 2 and image.shape[1] == 2:
        image, positions = positions, image
        warnings.warn("The order of arguments for optimap.show_positions() has changed.", DeprecationWarning)

    if image is not None:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        ax.axis("off")

    if isinstance(color, str) or color is None:
        color = [color, ] * len(positions)
    if isinstance(size, (int, float)) or size is None:
        size = [size, ] * len(positions)
    for pos, s, c in zip(positions, size, color):
        ax.scatter(pos[0], pos[1], s=s, c=c, **kwargs)
    if show:
        plt.show()
    return ax


def show_traces(traces, x=None, fps=None, colors=None, labels=None, ax=None, **kwargs):
    """Plot one or more traces.

    Parameters
    ----------
    traces : 1D or 2D array
        Traces to plot
    x : 1D array, optional
        X-axis values, by default None
    fps : float, optional
        Sampling rate of the traces (frames per second), by default None.
        If passed x-axis is shown in seconds.
        ``x`` and ``fps`` cannot be passed at the same time.
    colors : list of colors, optional
        Colors for the traces, by default None
    labels : list of str, optional
        Labels for the traces, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    kwargs : dict, optional
        Additional arguments to pass to :py:func:`matplotlib.pyplot.plot`

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    if traces is None:
        return ax
    elif traces.ndim == 1:
        traces = np.expand_dims(traces, 1)

    show_legend = labels is not None
    labels = labels or [None] * traces.shape[1]
    colors = colors or [None] * traces.shape[1]

    if x is not None and fps is not None:
        msg = "`x` and `fps` parameters cannot be passed at the same time"
        raise ValueError(msg)

    if fps is not None:
        x = np.arange(traces.shape[0]) / fps
        x_label = "Time [s]"
    elif x is not None:
        x_label = None
    else:
        x = np.arange(traces.shape[0])
        x_label = "Frame"

    for i in range(traces.shape[1]):
        ax.plot(x, traces[:, i], color=colors[i], label=labels[i], **kwargs)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity")
    if show_legend:
        ax.legend()

    if show:
        plt.show()
    return ax

def show_trace(*args, **kwargs):
    """Alias for :py:func:`show_traces`."""
    return show_traces(*args, **kwargs)


def collage_positions(positions, image_shape, ncols=6):
    """Correspondant to :func:`image.collage` but for positions. Collages the positions in the same way as the images would be collaged.

    `positions` is a list of list of tuples, i.e. one list of positions for each image. The function collages the positions in the same way as the images would be collaged and returns a list of tuples where the positions have been offset to the correct position in the collage. All images are assumed to have the same shape `image_shape`.

    Parameters
    ----------
    positions : list of arrays
        List of list of positions to collage
    image_shape : tuple
        Shape of the images where the positions are from
    ncols : int, optional
        Number of columns, by default 6

    Returns
    -------
    list of tuples
        Collaged positions
    """
    collage = []
    n = 0
    offset_x = 0
    offset_y = 0
    while n < len(positions):
        stop = min(ncols, len(positions))
        for i in range(n, n + stop):
            points = positions[i]
            points[:, 1] += offset_x
            points[:, 0] += offset_y
            offset_x += image_shape[0]
            collage.extend(points)
        n += stop
        offset_x = 0
        offset_y += image_shape[1]
    return collage
