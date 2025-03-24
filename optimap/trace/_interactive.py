import matplotlib.pyplot as plt
import numpy as np

from ..utils import _print, interactive_backend
from ._core import extract_traces, show_traces
from ._point_clicker import PointClicker
from ..image import show_image


@interactive_backend
def select_positions(image, title="Click to select positions", as_integers=True, **kwargs):
    """Interactive selection of positions on an image.
    Click on the image to select a position. Right click to remove a position. Close the window to finish.

    Parameters
    ----------
    image : 2D array
        Image to select positions from
    as_integers : bool, optional
        Return pixel coordinates if True, by default True
    kwargs : dict, optional
        Additional keyword arguments for :py:func:`optimap.show_image`

    Returns
    -------
    list of tuples
        List of selected positions
    """
    _print(
        "Click positions on the image, close the window to finish. Right click a point to remove it."
    )

    if image.ndim == 3:
        image = image[0]

    fig, ax = plt.subplots()
    show_image(image, ax=ax, **kwargs)
    ax.set_title(title)
    klicker = PointClicker(ax, as_integer=as_integers)
    plt.show(block=True)

    return klicker.get_positions()

@interactive_backend
def select_traces(video, size=5, ref_frame=0, x=None, fps=None):
    """Interactive selection/plotting of traces from a video. Click on the image to select a position.
    Right click to remove a position. Close the window to finish.

    Parameters
    ----------
    video : 3D np.ndarray
        Video to select traces from
    size : int
        Size parameter for trace
    ref_frame : int or np.ndarray
        Reference frame of the first video to show. If an integer, the frame is taken from the video. If an array, the array is used.
    x : 1D array, optional
        X-axis values, by default None. See :py:func:`show_traces` for details.
    fps : float, optional
        Frames per second, by default None. See :py:func:`show_traces` for details.

    Returns
    -------
    traces : 2D np.ndarray
        Traces of the selected positions
    positions : list of tuples
        List of selected positions
    """
    _print(
        "Click positions on the image, close the window to finish. Right click a point to remove it."
    )
    if isinstance(ref_frame, np.ndarray):
        image = ref_frame
    else:
        image = video[ref_frame]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # subplot 0: image
    axs[0].imshow(image, cmap="gray", interpolation="none")
    axs[0].set_title("Click to select positions")
    klicker = PointClicker(axs[0], as_integer=True)

    # subplot 1: traces
    def setup_trace_axis():
        axs[1].set_xlim(0, video.shape[0])
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Intensity")
    setup_trace_axis()

    def update():
        nonlocal axs, fig
        axs[1].clear()
        traces = extract_traces(video, klicker.get_positions(), size)
        show_traces(traces, ax=axs[1], x=x, fps=fps)
        fig.canvas.draw()

    klicker.on_point_added(lambda pos: update())
    klicker.on_point_removed(lambda pos, idx: update())
    plt.show(block=True)

    coords = klicker.get_positions()
    traces = extract_traces(video, coords, size)
    return traces, coords
