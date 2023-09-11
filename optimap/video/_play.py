import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.colors import Colormap

from ..utils import interactive_backend
from ._export import iter_alpha_blend_videos


def play(video, skip_frame=1, title="", vmin=None, vmax=None, cmap="gray", interval=10, **kwargs):
    """
    Simple video player based on matplotlib's animate function.

    See :py:func:`optimap.video.play2` for a player for two videos side-by-side, and :py:func:`optimap.video.playn` for a player for `n` videos side-by-side.

    .. note::
        Using Monochrome is an alternative to this function, which allows for more control over the video player.

        See :py:func:`monochrome.show_array` for more information.

        >>> import monochrome as mc
        >>> mc.show_array(video)

    Parameters
    ----------
    video : {t, x, y} np.ndarray
        Video to play.
    skip_frame : int, optional
        Skip every n-th frame, by default 1
    title : str, optional
        Title of the video, by default ""
    vmin : float, optional
        Minimum value for the colorbar, by default None
    vmax : float, optional
        Maximum value for the colorbar, by default None
    cmap : str, optional
        Colormap to use, by default "gray"
    interval : int, optional
        Delay between frames in ms, by default 10.
        This is not the actual framerate, but the delay between frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    return playn(
        [video],
        skip_frame=skip_frame,
        titles=[title],
        cmaps=[cmap],
        vmins=[vmin],
        vmaxs=[vmax],
        interval=interval,
        **kwargs,
    )


def play2(
    video1,
    video2,
    skip_frame=1,
    title1="video1",
    title2="video2",
    cmap1="gray",
    cmap2="gray",
    interval=10,
    **kwargs,
):
    """
    Video player for two videos side-by-side based on matplotlib's animate function.

    Parameters
    ----------
    video1 : {t, x, y} np.ndarray
        First video to play.
    video2 : {t, x, y} np.ndarray
        Second video to play.
    skip_frame : int, optional
        Skip every n-th frame, by default 1
    title1 : str, optional
        Title of the first video, by default "video1"
    title2 : str, optional
        Title of the second video, by default "video2"
    cmap1 : str, optional
        Colormap to use for the first video, by default "gray"
    cmap2 : str, optional
        Colormap to use for the second video, by default "gray"
    interval : int, optional
        Delay between frames in ms, by default 10. This is not the actual framerate, but the delay between frames.
    **kwargs
        passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    
    return playn([video1, video2], skip_frame=skip_frame, titles=[title1, title2], cmaps=[cmap1, cmap2], interval=interval, **kwargs)


@interactive_backend
def playn(videos, skip_frame=1, titles=None, cmaps="gray", vmins=None, vmaxs=None, interval=10, **kwargs):
    """
    Video player for `n` side-by-side videos based on matplotlib's animate function.

    Parameters
    ----------
    videos: list of {t, x, y} np.ndarray
        Videos to play.
    skip_frame : int, optional
        Skip every n-th frame, by default 1
    titles : list of str, optional
        Titles of the videos, by default None
    cmaps : str or list of str, optional
        Colormaps to use for the videos, by default "gray"
    vmins : float or list of floats, optional
        Minimum values for the colorbars, by default None
    vmaxs : float or list of floats, optional
        Maximum values for the colorbars, by default None
    interval : int, optional
        Delay between frames in ms, by default 10. This is not the actual framerate, but the delay between frames.
    **kwargs
        passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """

    n = len(videos)
    nt = videos[0].shape[0]
    for i in range(n):
        if videos[i].shape[0] < nt:
            raise ValueError("videos have to be same length!")
        videos[i] = videos[i][::skip_frame]

    if titles is None:
        titles = [f"Video {i}" for i in range(n)]
    if isinstance(cmaps, str) or isinstance(cmaps, Colormap):
        cmaps = [cmaps for i in range(n)]

    if vmins is None:
        vmins = [np.min(videos[i][0]) for i in range(n)]
    elif isinstance(vmins, (int, float)):
        vmins = [vmins for _ in range(n)]
    if vmaxs is None:
        vmaxs = [np.max(videos[i][0]) for i in range(n)]
    elif isinstance(vmaxs, (int, float)):
        vmaxs = [vmaxs for _ in range(n)]

    fig, axs = plt.subplots(1, n, **kwargs)
    if n == 1:
        axs = [axs]

    suptitle = fig.suptitle(f"Frame {0:4d}", font="monospace")
    imshows = []
    for i in range(n):
        imshows.append(
            axs[i].imshow(videos[i][0], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
        )
        axs[i].set_title(f"{titles[i]}")
        axs[i].axis("off")
    fig.tight_layout()

    def update(frame):
        for i in range(n):
            imshows[i].set_data(videos[i][frame])
            suptitle.set_text(f"Frame {frame*skip_frame:4d}")

    ani = animation.FuncAnimation(
        fig, update, frames=videos[0].shape[0], interval=interval
    )
    plt.show(block=True)
    return ani


@interactive_backend
def play_with_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: np.ndarray = None,
    skip_frames: int = 1,
    cmap_base="gray",
    cmap_overlay="Purples",
    vmin_base=None,
    vmax_base=None,
    vmin_overlay=None,
    vmax_overlay=None,
    interval=10,
    **kwargs,
):
    fig, ax = plt.subplots(**kwargs)

    suptitle = fig.suptitle(f"Frame {0:4d}", font="monospace")

    imshow = ax.imshow(base[0])
    ax.axis("off")
    fig.tight_layout()

    def update(frame):
        i, frame = frame
        imshow.set_data(frame)
        suptitle.set_text(f"Frame {i*skip_frames:4d}")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=enumerate(iter_alpha_blend_videos(
            base=base,
            overlay=overlay,
            alpha=alpha,
            skip_frames=skip_frames,
            cmap_base=cmap_base,
            cmap_overlay=cmap_overlay,
            vmin_base=vmin_base,
            vmax_base=vmax_base,
            vmin_overlay=vmin_overlay,
            vmax_overlay=vmax_overlay,
        )),
        interval=interval,
        save_count=base.shape[0],
    )
    plt.show(block=True)
    return ani
