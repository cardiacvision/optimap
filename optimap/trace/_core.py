import numpy as np
import matplotlib.pyplot as plt

from ..image import disc_mask

TRACE_TYPE = "rect"


def set_default_trace_window(window):
    """
    Set the default trace window type.

    Parameters
    ----------
    window : str
        Type of trace, one of 'rect', 'disc', 'pixel'
        See :py:func:`extract_trace` for more information.
    """
    if window not in ["rect", "disc", "pixel"]:
        raise ValueError(f"Unknown trace type {window}")

    global TRACE_TYPE
    TRACE_TYPE = window


def get_default_trace_window():
    """
    Get the default trace window.

    Returns
    -------
    str
        Type of trace, one of 'rect', 'disc', 'pixel'
    """
    return TRACE_TYPE


def extract_traces(video, coords, size=5, show=False, window=None, **kwargs):
    """
    Extract trace or traces from a video at specified coordinates.
    
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
    type : str, optional
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
    """
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
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

    def slice(video, x, y):
        xs, xe = max(0, x - size // 2), min(video.shape[1], x + size // 2 + size % 2)
        ys, ye = max(0, y - size // 2), min(video.shape[2], y + size // 2 + size % 2)

        return video[:, xs:xe, ys:ye]

    if size == 0 or size == 1 or window == "pixel":
        traces = [video[:, x, y] for (y, x) in coords]
    elif window == "rect":
        traces = [slice(video, x, y).mean(axis=(1, 2)) for (y, x) in coords]
    elif window == "disc":
        traces = [
            video[:, disc_mask(video.shape[1:], (x, y), size / 2)].mean(axis=(1,))
            for (y, x) in coords
        ]
    else:
        raise ValueError(f"Unknown trace window type '{window}'")
    traces = np.array(traces).T

    if single_coord:
        traces = traces[..., 0]

    if show:
        show_traces(traces, **kwargs)
    return traces


def show_positions(image, positions, ax=None):
    """
    Overlay positions on an image.

    Parameters
    ----------
    image : 2D array
    positions : list of tuples
        List of positions to overlay
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    ax.imshow(image, cmap="gray", interpolation="none")
    ax.axis('off')
    for pos in positions:
        ax.scatter(pos[0], pos[1])
    if show:
        plt.show()
    return ax


def show_traces(traces, x=None, fps=None, colors=None, labels=None, ax=None, **kwargs):
    """
    Plot one or more traces.

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
        raise ValueError("`x` and `fps` parameters cannot be passed at the same time")
    
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
    """
    Alias for :py:func:`show_traces`.
    """
    return show_traces(*args, **kwargs)