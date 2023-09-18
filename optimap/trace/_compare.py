import numpy as np
import matplotlib.pyplot as plt

from ._core import extract_traces
from ._point_clicker import PointClicker

from ..utils import interactive_backend

@interactive_backend
def _compare_traces_interactive(videos, labels=None, size=5, ref_frame=0, colors=None, x=None, x_label=None):
    """
    Compare traces of multiple videos interactively. Click on the image to select a position. Close the window to finish.
    """
    image = videos[0][ref_frame]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # subplot 0: image
    axs[0].imshow(image, cmap="gray", interpolation="none")
    axs[0].set_title("Click to select points")
    klicker = PointClicker(axs[0], as_integer=True, single_point=True)

    # subplot 1: traces
    def on_point_added(pos):
        axs[1].clear()
        axs[1].set_xlim(x[0], x[-1])
        traces = []
        for video in videos:
            trace = extract_traces(video, pos, size)
            traces.append(trace)
        traces = np.array(traces).T
        for i in range(traces.shape[1]):
            axs[1].plot(x, traces[:, i], color=colors[i])
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel("Intensity")
        if labels:
            axs[1].legend(labels, ncols=len(labels))
        fig.canvas.draw()

    klicker.on_point_added(on_point_added)
    plt.show(block=True)

def _compare_traces_plot(videos, coords, size=5, labels=None, colors=None, x=None, x_label=None):
    """
    Plot traces from multiple videos at given coordinates.
    """
    if len(coords) == 0:
        print("No coordinates given")
        return
    elif len(coords) == 2 and isinstance(coords[0], int):
        coords = [coords]

    all_traces = []
    for coord in coords:
        traces = []
        for video in videos:
            traces.append(extract_traces(video, coord, size))
        traces = np.array(traces).T
        all_traces.append(traces)
    
    fig, ax = plt.subplots(nrows=len(coords), ncols=1, sharex=True, figsize=(5, 2.5*len(coords)))
    if len(coords) == 1:
        ax = [ax]
    for i, traces in enumerate(all_traces):
        for j in range(traces.shape[1]):
            ax[i].plot(x, traces[:, j], color=colors[j])
        ax[i].set_title(f"({coords[i][0]}, {coords[i][1]})")
        # ax[i].set_ylabel("Intensity")
    ax[-1].set_xlabel(x_label)
    if labels:
        fig.legend(labels, loc='outside upper center', ncols=len(labels))
    plt.xlim(x[0], x[-1])
    plt.show()

def compare_traces(videos, coords=None, labels=None, colors=None, size=5, ref_frame=0, fps=None, x=None):
    """
    Compare traces of multiple videos. If ``coords`` is given, traces are plotted at the given coordinates. Otherwise, an interactive window is opened to select coordinates by clicking on the image. Close the window to finish.

    Parameters
    ----------
    videos : list of 3D arrays
        List of videos to compare
    coords : tuple or list of tuples
        Coordinates of the traces
    labels : list of strings, optional
        List of labels for the videos, by default None
    colors : list of colors, optional
        Colors for the traces, has to be the same length as videos, by default None
    size : int, optional
        Size parameter of the trace, by default 5
    ref_frame : int, optional
        Reference frame of the first video to show, by default 0
        Only used if coords is None
    fps : float, optional
        Sampling rate of the traces (frames per second), by default None.
        If passed x-axis is shown in seconds.
        ``x`` and ``fps`` cannot be passed at the same time.
    x : 1D array, optional
        X-axis values, by default None
    """
    colors = colors or [None] * len(videos)
    if x is not None and fps is not None:
        raise ValueError("`x` and `fps` parameters cannot be passed at the same time")
    
    if fps is not None:
        x = np.arange(videos[0].shape[0]) / fps
        x_label = "Time [s]"
    elif x is not None:
        x_label = None
    else:
        x = np.arange(videos[0].shape[0])
        x_label = "Frame"

    if coords is None:
        return _compare_traces_interactive(videos, labels=labels, size=size, ref_frame=ref_frame, colors=colors, x=x, x_label=x_label)
    else:
        return _compare_traces_plot(videos, coords, size=size, labels=labels, colors=colors, x=x, x_label=x_label)
