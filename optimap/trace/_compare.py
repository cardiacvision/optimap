import numpy as np
import matplotlib.pyplot as plt

from ._core import extract_traces
from ._point_clicker import PointClicker

from ..utils import interactive_backend

@interactive_backend
def _compare_traces_interactive(videos, labels=None, size=5, ref_frame=0):
    """
    Compare traces of multiple videos interactively. Click on the image to select a position. Close the window to finish.

    Parameters
    ----------
    videos : list of 3D arrays
        List of videos to compare
    labels : list of strings, optional
        List of labels for the videos, by default None
    size : int, optional
        Size parameter of the trace, by default 5
    ref_frame : int, optional
        Reference frame of the first video to show, by default 0
        Only used if coords is None
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
        axs[1].set_xlim(0, videos[0].shape[0])
        traces = []
        for video in videos:
            trace = extract_traces(video, pos, size)
            traces.append(trace)
        traces = np.array(traces).T
        axs[1].plot(traces, label=labels)
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Intensity")
        if labels:
            axs[1].legend()
        fig.canvas.draw()

    klicker.on_point_added(on_point_added)
    plt.show(block=True)

def _compare_traces_plot(videos, coords, size=5, labels=None):
    """
    Plot traces from multiple videos at given coordinates.

    Parameters
    ----------
    videos : list of 3D arrays
        Videos to extract traces from
    coords : tuple or list of tuples
        Coordinates of the traces
    size : int, optional
        Size parameter for selection mask, by default 5px.
    labels : list of str, optional
        Labels for the videos, by default None
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
        ax[i].plot(traces)
        ax[i].set_title(f"({coords[i][0]}, {coords[i][1]})")
        # ax[i].set_ylabel("Intensity")
    ax[-1].set_xlabel("Frame")
    if labels:
        fig.legend(labels, loc='outside lower center', ncols=len(labels))
    plt.xlim(0, videos[0].shape[0])
    plt.show()

def compare_traces(videos, coords=None, labels=None, size=5, ref_frame=0):
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
    size : int, optional
        Size parameter of the trace, by default 5
    ref_frame : int, optional
        Reference frame of the first video to show, by default 0
        Only used if coords is None
    """

    if coords is None:
        return _compare_traces_interactive(videos, labels=labels, size=size, ref_frame=ref_frame)
    else:
        return _compare_traces_plot(videos, coords, size=size, labels=labels)
