import matplotlib.pyplot as plt
import numpy as np

from ..image import show_image
from ..utils import _print


def find_crossings(signal, threshold=0.5, inverted=False, min_duration=8):
    """
    Finds the indices where a 1D signal crosses a threshold from below to above (`inverted=False`) or from above to below (`inverted=True`).
    This is useful for detecting events in time series data.

    Parameters
    ----------
        signal: np.ndarray
            The 1D signal.
        threshold: float
            The threshold value.
        inverted: bool
            If True, find crossings from above to below the threshold.
        min_duration: int
            Minimum duration of the crossing in frames. If set to 0, all crossings are returned.

    Returns
    -------
        np.ndarray: An array of indices where the signal crosses the threshold in the specified direction.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-dimensional")
    above_threshold = signal > threshold
    change = -1 if inverted else 1
    crossing_indices = np.where(np.diff(above_threshold.astype(int)) == change)[0]
    def crossing_filter(idx):
        if idx - min_duration // 2 < 0 or idx + min_duration // 2 >= len(signal):
            return False
        if inverted:
            if (not above_threshold[idx - min_duration // 2]) and above_threshold[idx + min_duration // 2]:
                return True
        else:
            if above_threshold[idx - min_duration // 2] and (not above_threshold[idx + min_duration // 2]):
                return True
        return False
    if min_duration > 0:
        # filter crossings based on duration
        crossing_indices = np.array([crossing for crossing in crossing_indices if crossing_filter(crossing)])
    return crossing_indices


def show_activation_map(activation_map,
                        vmin=0,
                        vmax=None,
                        title="",
                        cmap="turbo",
                        show_contour=True,
                        show_colorbar=True,
                        colorbar_title="Activation Time",
                        ax=None,
                        contour_fmt=None,
                        contour_levels=None,
                        contour_fontsize=None,
                        contour_linestyles=None,
                        contour_args={},
                        contour_label_args={}):
    """Display an activation/isochrone map with optional contours.
    
    This function visualizes activation maps (also known as isochrone maps) which show the timing 
    of activation across a 2D spatial region. It can display the activation map as a color-coded image 
    with optional contour lines to highlight isochrones.
    
    Parameters
    ----------
    activation_map : 2D ndarray
        The activation map to display. Values represent activation times.
    vmin : float, optional
        Minimum value for the colormap, by default 0
    vmax : float, optional
        Maximum value for the colormap, by default None (auto-determined from data)
    cmap : str, optional
        Colormap to use for the activation map, by default "turbo"
    title : str, optional
        Title for the plot, by default ""
    show_contour : bool, optional
        Whether to overlay contour lines on the activation map, by default True
    show_colorbar : bool, optional
        Whether to display a colorbar, by default True
    colorbar_title : str, optional
        Title for the colorbar, by default "Activation Time"
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes is created, by default None
    contour_fontsize : int or float, optional
        Font size for contour labels, by default None (auto-determined)
    contour_fmt : str, optional
        Format string for contour labels e.g. `' %1.0f ms '`, by default None. See :func:`matplotlib.pyplot.clabel` for details.
    contour_levels : array-like, optional
        Specific contour levels to draw, by default None (auto-determined). See :func:`matplotlib.pyplot.contour` for details.
    contour_linestyles : str or list of str, optional
        Line style(s) for contour lines, by default None. See :func:`matplotlib.pyplot.contour` for details.
    contour_args : dict, optional
        Additional keyword arguments for matplotlib's contour function, by default {}
    contour_label_args : dict, optional
        Additional keyword arguments for contour label formatting, by default {}
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    
    See Also
    --------
    compute_activation_map : For creating activation maps from video data
    """
    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False
    show_image(activation_map, vmin=vmin, vmax=vmax, cmap=cmap, title=title, show_colorbar=show_colorbar, colorbar_title=colorbar_title, ax=ax)

    if show_contour:
        contours = ax.contour(activation_map, levels=contour_levels, linestyles=contour_linestyles, colors="black", **contour_args)

        contours.clabel(fontsize=contour_fontsize, fmt=contour_fmt, **contour_label_args)
    if show:
        plt.show()
    return ax

def compute_activation_map(video,
                           threshold=0.5,
                           inverted=False,
                           fps=None,
                           set_nan_for_inactive=True,
                           show=True,
                           **kwargs):
    """Computes an activation map (or isochrone map) from a given video based on pixel intensity thresholding.

    For each pixel in the video, the function determines the time (or frame index) at which the pixel's intensity
    first surpasses (or falls below, if inverted is set to True) the specified threshold.

    If `fps` is specified, time is giving in milliseconds, otherwise, it is given in frames.

    Parameters
    ----------
    video : np.ndarray
        A 3D array representing the video, with dimensions {t (time or frames), x (width), y (height)}.
    threshold : float, optional
        Intensity threshold at which a pixel is considered activated. Defaults to 0.5.
    inverted : bool, optional
        If True, the function will compute the time/frame when pixel intensity falls below the threshold,
        rather than surpassing it. Defaults to False.
    fps : float, optional
        If provided, the resulting activation map will represent times in milliseconds based on this frame rate,
        otherwise, it will be in frames.
    set_nan_for_inactive : bool, optional
        If True, pixels that never reach the activation threshold will be set to NaN. Defaults to True.
    show : bool, optional
        If True, the resulting activation map will be displayed. Defaults to True.
    cmap : str, optional
        Colormap to use for displaying the activation map. Defaults to 'jet'.

    Returns
    -------
    activation_map : ndarray
        2D image
    """
    _print(f"computing activation map with {threshold=}")
    if video.ndim != 3:
        msg = "video must be 3-dimensional"
        raise ValueError(msg)

    if inverted:
        amap = np.nanargmax(video < threshold, axis=0)
    else:
        amap = np.nanargmax(video > threshold, axis=0)

    amap = amap.astype(np.float32)
    if fps is not None:
        amap = amap * 1000.0 / fps

    if set_nan_for_inactive:
        # set all pixels to NaN that never reach threshold
        if inverted:
            never_activated = (amap == 0) & (video[0] > threshold)
        else:
            never_activated = (amap == 0) & (video[0] < threshold)
        amap[never_activated] = np.nan

    # set masked pixels to NaN
    amap[np.isnan(video[0])] = np.nan

    _print(f"minimum of activation_map: {np.nanmin(amap)}")
    _print(f"maximum of activation_map: {np.nanmax(amap)}")

    if show:
        cbar_label = "Activation Time [ms]" if fps is not None else "Activation Time [frames]"
        contour_fmt = " %1.0f ms " if fps is not None else " %1.0f frames "
        title="Activation Map"
        show_activation_map(amap, contour=True, title=title,colorbar_title=cbar_label,  contour_fmt=contour_fmt, **kwargs)
    return amap
