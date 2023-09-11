import numpy as np
import matplotlib.pyplot as plt

from ._core import extract_traces
from ._point_clicker import PointClicker
from ..utils import _print, print_bar

from ..utils import interactive_backend

@interactive_backend
def select_positions(image, as_integers=True):
    """
    Interactive selection of positions on an image.
    Click on the image to select a position. Right click to remove a position. Close the window to finish.

    Parameters
    ----------
    image : 2D array
    as_integers : bool, optional
        Return pixel coordinates if True, by default True

    Returns
    -------
    list of tuples
        List of selected positions
    """
    _print(
        f"Click positions on the image, close the window to finish. Right click a point to remove it."
    )

    if image.ndim == 3:
        image = image[0]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title("Click to select positions")
    klicker = PointClicker(ax, as_integer=as_integers)
    plt.show(block=True)

    coords = klicker.get_positions()
    return coords

@interactive_backend
def select_traces(video, size=5, ref_frame=0):
    """
    Interactive selection/plotting of traces from a video. Click on the image to select a position. Right click to remove a position. Close the window to finish.

    Parameters
    ----------
    video : 3D np.ndarray
        Video to select traces from
    size : int
        Size parameter for trace
    ref_frame : int
        Reference frame of the first video to show

    Returns
    -------
    traces : 2D np.ndarray
        Traces of the selected positions
    positions : list of tuples
        List of selected positions
    """
    _print(
        f"Click positions on the image, close the window to finish. Right click a point to remove it."
    )

    image = video[ref_frame]
    coords = []
    traces = None

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

    def on_point_added(pos):
        nonlocal coords, traces, axs, fig
        trace = extract_traces(video, [pos], size)
        axs[1].plot(trace)
        fig.canvas.draw()

        coords.append(pos)
        if traces is None:
            traces = trace
        else:
            traces = np.hstack((traces, trace))

    def on_point_removed(pos, idx):
        nonlocal coords, traces, axs, fig
        coords = klicker.get_positions()
        traces = extract_traces(video, coords, size)
        axs[1].clear()
        setup_trace_axis()
        axs[1].plot(traces)
        fig.canvas.draw()

    klicker.on_point_added(on_point_added)
    klicker.on_point_removed(on_point_removed)
    plt.show(block=True)

    # coords = klicker.get_positions()
    # traces = extract_trace_values(video, coords, size)
    return traces, coords
