import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ..utils import _print, interactive_backend


@interactive_backend
def play_displacements(
    video, vectors, skip_frame=1, vskip=5, vcolor="red", vscale=1.0, title="video"
):
    # TODO: fix dead vectors when displaying cropped videos / vector fields
    """Simple video player with displacement vectors displayed as arrows.

    Parameters
    ----------
    video : {t, x, y} ndarray
        video to play
    vectors : {t, x, y, 2} ndarray
        Optical flow / displacement vectors.
    skip_frame : int
        only show every n-th frame
    vskip : int
        shows only every nth vector
    vcolor : str
        color of the vectors
    vscale : float
        scales the displayed vectors
    title : str
        title of the plot

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _print(f"playing video with displacement vectors ({skip_frame=}, {vskip=})... ")
    video = video[::skip_frame]
    vmax = np.nanmax(video)
    vmin = np.nanmin(video)
    vectors = vectors[::skip_frame, ::vskip, ::vskip]
    vectors[..., 0] /= video.shape[1]
    vectors[..., 1] /= video.shape[2]
    X = np.arange(0, video.shape[1], vskip)
    Y = np.arange(0, video.shape[2], vskip)
    fig, ax = plt.subplots()

    def update(frame):
        ux = vectors[frame, :, :, 0]
        uy = vectors[frame, :, :, 1] * -1.0
        ax.clear()
        ax.imshow(video[frame], cmap="gray", vmin=vmin, vmax=vmax)
        ax.quiver(X, Y, ux, uy, scale=vscale, color=vcolor)
        # ax.quiver(ux, uy, scale=vscale, color=vcolor)
        ax.set_title(title + "\n frame {}".format(frame * skip_frame))

    ani = animation.FuncAnimation(fig, update, frames=video.shape[0], interval=50)
    plt.show(block=True)
    return ani

@interactive_backend
def play_displacements_points(
    video, vectors, skip_frame=1, vskip=10, vcolor="black", psize=5, title=""
):
    """Simple video player with displacement vectors displayed as points.

    Parameters
    ----------
    video : {t, x, y} ndarray
        video to play
    vectors : {t, x, y, 2} ndarray
        optical flow / displacement vectors.
    skip_frame : int
        only show every n-th frame
    vskip : int
        shows only every nth vector
    vcolor : str
        color of the vectors, default: black
    psize : int
        size of the points, default: 5
    title : str
        title of the plot

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    # TO DO!!! fix bug with vskip, now works only with 5, 10
    if video.shape[1] % vskip != 0 or video.shape[2] % vskip != 0:
        msg = "image size must be divisible by vskip!"
        raise ValueError(msg)

    _print(f"playing video with displacement vectors ({skip_frame=}, {vskip=}...")
    video = video[::skip_frame]
    vmax = np.nanmax(video)
    vmin = np.nanmin(video)
    vectors = vectors[::skip_frame, ::vskip, ::vskip]
    vectors[..., 0] /= video.shape[1]
    vectors[..., 1] /= video.shape[2]
    X = np.arange(0, video.shape[1], vskip)
    Y = np.arange(0, video.shape[2], vskip)
    PX, PY = np.meshgrid(X, Y)

    fig, ax = plt.subplots()

    def update(frame):
        ux = vectors[frame, :, :, 0]
        uy = vectors[frame, :, :, 1]
        px = PX + ux
        py = PY + uy
        ax.clear()
        ax.imshow(video[frame], cmap="gray", vmin=vmin, vmax=vmax)
        ax.scatter(px, py, color=vcolor, marker="o", s=psize)
        ax.set_xlim(1, video.shape[1])
        ax.set_ylim(video.shape[2], 1)
        ax.set_title(title + "\n frame {}".format(frame * skip_frame))

    ani = animation.FuncAnimation(fig, update, frames=video.shape[0], interval=50)
    plt.show(block=True)
    return ani
