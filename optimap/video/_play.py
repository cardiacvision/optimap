import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Colormap

from ..utils import deprecated, interactive_backend
from ._export import iter_alpha_blend_videos
from ._player import InteractivePlayer

@deprecated("use optimap.show_video instead")
def play(video, skip_frame=1, title="", vmin=None, vmax=None, cmap="gray", interval=10, **kwargs):
    """Deprecated alias for :py:func:`optimap.video.show_video`."""
    return show_video(video, skip_frame=skip_frame, title=title, vmin=vmin, vmax=vmax,
                      cmap=cmap, interval=interval, **kwargs)


@deprecated("use optimap.show_video_pair instead")
def play2(video1, video2, skip_frame=1, title1="", title2="", vmin1=None, vmax1=None, vmin2=None, vmax2=None, cmap1="gray", cmap2="gray", interval=10, **kwargs):
    """Deprecated alias for :py:func:`optimap.video.show_video_pair`."""
    return show_video_pair(video1, video2, skip_frame=skip_frame, title1=title1, title2=title2,
                           vmin1=vmin1, vmax1=vmax1, vmin2=vmin2, vmax2=vmax2, cmap1=cmap1, cmap2=cmap2,
                           interval=interval, **kwargs)


@deprecated("use optimap.show_videos instead")
def playn(videos, skip_frame=1, titles=None, cmaps="gray", vmins=None, vmaxs=None, interval=10, **kwargs):
    """Deprecated alias for :py:func:`optimap.video.show_videos`."""
    return show_videos(videos, skip_frame=skip_frame, titles=titles, cmaps=cmaps, vmins=vmins, vmaxs=vmaxs, interval=interval, **kwargs)


@deprecated("use optimap.show_videos instead")
def play_with_overlay(*args, **kwargs):
    """Deprecated alias for :py:func:`optimap.video.show_video_overlay`."""
    return show_video_overlay(*args, **kwargs)


def show_video(video, skip_frame=1, title="", vmin=None, vmax=None, cmap="gray", interval=10, **kwargs):
    """Simple video player based on matplotlib's animate function.

    See :py:func:`optimap.video.show_video_pair` for a player for two videos side-by-side, and :py:func:`optimap.video.show_videos`
    for a player for `n` videos side-by-side.

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
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    InteractivePlayer
    """
    return show_videos(
        [video],
        skip_frame=skip_frame,
        titles=[title],
        cmaps=[cmap],
        vmins=[vmin],
        vmaxs=[vmax],
        interval=interval,
        **kwargs,
    )


def show_video_pair(
    video1,
    video2,
    skip_frame=1,
    title1="video1",
    title2="video2",
    cmap1="gray",
    cmap2="gray",
    interval=10,
    vmin1=None,
    vmax1=None,
    vmin2=None,
    vmax2=None,
    **kwargs,
):
    """Video player for two videos side-by-side based on matplotlib's animate function.

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
    vmin1 : float, optional
        Minimum value for the colorbar of the first video, by default None
    vmax1 : float, optional
        Maximum value for the colorbar of the first video, by default None
    vmin2 : float, optional
        Minimum value for the colorbar of the second video, by default None
    vmax2 : float, optional
        Maximum value for the colorbar of the second video, by default None
    interval : int, optional
        Delay between frames in ms, by default 10. This is not the actual framerate, but the delay between frames.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    InteractivePlayer
    """
    if vmin1 is None and vmin2 is None:
        vmins = None
    else:
        vmins = [vmin1, vmin2]
    if vmax1 is None and vmax2 is None:
        vmaxs = None
    else:
        vmaxs = [vmax1, vmax2]
    return show_videos([video1, video2],
                 skip_frame=skip_frame,
                 titles=[title1, title2],
                 cmaps=[cmap1, cmap2],
                 vmins=vmins,
                 vmaxs=vmaxs,
                 interval=interval,
                 **kwargs)


@interactive_backend
def show_videos(videos, skip_frame=1, titles=None, cmaps="gray", vmins=None, vmaxs=None, interval=10, **kwargs):
    """Video player for `n` side-by-side videos based on matplotlib's animate function.

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
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    InteractivePlayer
    """
    n = len(videos)
    nt = len(videos[0])
    for i in range(n):
        if videos[i].ndim < 3 or videos[i].ndim > 4:
            msg = f"Video {i} is not a video, it has shape {videos[i].shape}! Use show_image or show_traces instead."
            raise ValueError(msg)
        if len(videos[i]) < nt:
            msg = "videos have to be same length!"
            raise ValueError(msg)

    if titles is None:
        titles = [f"Video {i}" for i in range(n)]
    if isinstance(cmaps, (Colormap, str)):
        cmaps = [cmaps for i in range(n)]

    if vmins is None:
        vmins = [np.nanmin(videos[i][0]) for i in range(n)]
    elif isinstance(vmins, (int, float)):
        vmins = [vmins for _ in range(n)]
    if vmaxs is None:
        vmaxs = [np.nanmax(videos[i][0]) for i in range(n)]
    elif isinstance(vmaxs, (int, float)):
        vmaxs = [vmaxs for _ in range(n)]

    fig, axs = plt.subplots(1, n, **kwargs)
    if n == 1:
        axs = [axs]

    # needed to set the correct spacing between GUI elements for Player
    fig.suptitle("  ", font="monospace")

    imshows = []
    for i in range(n):
        imshows.append(
            axs[i].imshow(videos[i][0], cmap=cmaps[i], interpolation="none", vmin=vmins[i], vmax=vmaxs[i])
        )
        axs[i].set_title(f"{titles[i]}")
        axs[i].axis("off")
    fig.tight_layout()

    def update(i):
        for j in range(n):
            imshows[j].set_data(videos[j][i])

    ani = InteractivePlayer(
        fig=fig,
        func=update,
        start=0,
        end=len(videos[0]),
        step=skip_frame,
        interval=interval,
    )
    plt.show(block=True)
    return ani


@interactive_backend
def show_video_overlay(
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
    """Play a video with an overlay. See :func:`export_video_with_overlay` and :func:`iter_alpha_blend_videos` for more information.

    Parameters
    ----------
    base : np.ndarray
        Base video to play.
    overlay : np.ndarray
        Video to overlay on the base video.
    alpha : np.ndarray, optional
        Alpha channel for the overlay, by default None
    skip_frames : int, optional
        Show every n-th frame, by default 1
    cmap_base : str or matplotlib.colors.Colormap, optional
        Colormap to use for the base video, by default "gray"
    cmap_overlay : str or matplotlib.colors.Colormap, optional
        Colormap to use for the overlay video, by default "Purples"
    vmin_base : float, optional
        Minimum value for the colorbar of the base video, by default None
    vmax_base : float, optional
        Maximum value for the colorbar of the base video, by default None
    vmin_overlay : float, optional
        Minimum value for the colorbar of the overlay video, by default None
    vmax_overlay : float, optional
        Maximum value for the colorbar of the overlay video, by default None
    interval : int, optional
        Delay between frames in ms, by default 10. This is not the actual framerate, but the delay between frames.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(**kwargs)

    suptitle = fig.suptitle(f"Frame {0:4d}", font="monospace")

    imshow = ax.imshow(base[0], vmin=0, vmax=1)
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
