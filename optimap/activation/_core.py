import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..image import show_image
from ..trace import show_traces
from ..utils import _print


def show_activations(signal, activations, fps=None, ax=None, linecolor="red", linestyle="--", **kwargs):
    """Display a signal with vertical lines marking activation times.
    
    This function plots a 1D signal and adds vertical lines at specified activation points,
    which is useful for visualizing detected events or activation times in time series data.
    
    Parameters
    ----------
    signal : array-like
        The 1D signal to display
    activations : array-like
        List of activation points (frame indices)
    fps : float, optional
        Frames per second for time conversion. If provided, x-axis will show time in seconds 
        instead of frame numbers.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure and axes is created.
    linecolor : str, optional
        Color of the vertical lines marking activation times, by default "red"
    linestyle : str, optional
        Line style of the vertical lines marking activation times, by default "--"
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_traces`.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
        
    See Also
    --------
    find_activations : For detecting activation times in signals
    """
    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False
    ax = show_traces(signal, fps=fps, ax=ax, **kwargs)
    fps = fps if fps is not None else 1
    for activation in activations:
        ax.axvline(activation / fps, color=linecolor, linestyle=linestyle)
    if show:
        plt.show()
    return ax


def find_activations(signal, threshold=0.5, interpolate=False, falling_edge=False, min_duration=8, fps=None, show=True, ax=None):
    """
    Finds the frame indices where a 1D signal crosses a threshold from below to above (`falling_edge=False`) or from above to below (`falling_edge=True`).
    
    This is useful for computing local activation times or detecting events in a time series.

    By default, the function returns the **closest** frame index where the signal crosses the threshold. If `interpolate` is set to True, the function will return the exact crossing point using linear interpolation.

    Parameters
    ----------
        signal: np.ndarray
            The 1D signal.
        threshold: float
            The threshold value.
        interpolate: bool
            If True, use linear interpolation to find the exact crossing point.
        falling_edge: bool
            If True, find crossings from above to below the threshold.
        min_duration: int
            Minimum duration of the crossing in frames. If set to 0, all crossings are returned.

    Returns
    -------
        np.ndarray: An array of indices where the signal crosses the threshold in the specified direction.
    """
    if signal.ndim == 3:
        # mean signal over the spatial dimensions
        signal = np.nanmean(signal, axis=(1, 2))
    elif signal.ndim == 2:
        # signal is a trace (T, N)
        return [
            find_activations(signal[:, i], threshold=threshold, interpolate=interpolate, min_duration=min_duration, falling_edge=falling_edge, show=False) for i in range(signal.shape[1])
        ]
    elif signal.ndim != 1:
        raise ValueError("Error: signal is not a video or trace.")
    
    if falling_edge:
        condition = signal > threshold
    else:
        condition = signal < threshold
    crossing_indices = np.where(np.diff(condition.astype(int)) == -1)[0]

    if min_duration > 0:
        # filter crossings based on duration
        def crossing_filter(idx):
            for i in range(idx - min_duration // 2, idx):
                if i >= 0 and not condition[i]:
                    return False
            for i in range(idx + 2, idx + min_duration // 2 + 2):
                if i < len(signal) and condition[i]:
                    return False
            return True
        crossing_indices = np.array([crossing for crossing in crossing_indices if crossing_filter(crossing)])

    if len(crossing_indices) > 0:
        # linear interpolation to find the exact crossing point
        crossing_indices = crossing_indices + (threshold - signal[crossing_indices]) / (signal[crossing_indices + 1] - signal[crossing_indices])

    if not interpolate:
        crossing_indices = np.round(crossing_indices).astype(int)

    if show:
        show_activations(signal, crossing_indices, fps=fps, ax=ax)
    return crossing_indices


def show_activation_map(activation_map,
                        vmin=0,
                        vmax=None,
                        fps=None,
                        cmap="turbo",
                        title="",
                        show_contours=False,
                        show_map=True,
                        show_colorbar=True,
                        colorbar_title=None,
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
    fps : float, optional
        Show activation map times in milliseconds based on this frame rate, otherwise, it will be in frames.
    cmap : str, optional
        Colormap to use for the activation map, by default "turbo"
    title : str, optional
        Title for the plot, by default ""
    show_contours : bool, optional
        Whether to overlay contour lines on the activation map, by default True
    show_map : bool, optional
        Whether to display the activation map, by default True
    show_colorbar : bool, optional
        Whether to display a colorbar, by default True
    colorbar_title : str, optional
        Title for the colorbar, by default "Activation Time [ms]" if `fps` is provided, otherwise "Activation Time [frames]"
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

    if fps is not None:
        # Convert from frames to milliseconds
        activation_map = activation_map * 1000.0 / fps
        colorbar_title = "Activation Time [ms]" if colorbar_title is None else colorbar_title
        contour_fmt = " %1.0f ms " if contour_fmt is None else contour_fmt
    else:
        colorbar_title = "Activation Time [frames]" if colorbar_title is None else colorbar_title

    if show_map:
        show_image(activation_map, vmin=vmin, vmax=vmax, cmap=cmap, title=title, show_colorbar=show_colorbar, colorbar_title=colorbar_title, ax=ax)

    if show_contours:
        contours_kwargs = {"levels": contour_levels, "linestyles": contour_linestyles, "colors": "black"}
        contours_kwargs.update(contour_args)
        contours = ax.contour(activation_map, **contours_kwargs)

        contour_label_kwargs = {"fontsize": contour_fontsize, "fmt": contour_fmt}
        contour_label_kwargs.update(contour_label_args)
        contours.clabel(**contour_label_kwargs)
    if show:
        plt.show()
    return ax

def compute_activation_map(video,
                           threshold=0.5,
                           falling_edge=False,
                           interpolate=False,
                           min_duration=0,
                           normalize_time=True,
                           set_nan_for_inactive=True,
                           show=True,
                           **kwargs):
    """Computes an activation map (or isochrone map) from a given video based on pixel intensity thresholding.

    For each pixel in the video, the function determines the time (or frame index) at which the pixel's intensity
    first surpasses (or falls below, if `falling_edge` is set to True) the specified threshold.

    The activation map is given in terms of frames, if `interpolate=True` fractions of frames are returned. See :func:`show_activation_map` for plotting.

    Parameters
    ----------
    video : np.ndarray
        A 3D array representing the video, with dimensions {t (time or frames), x (width), y (height)}.
    threshold : float, optional
        Intensity threshold at which a pixel is considered activated. Defaults to 0.5.
    falling_edge : bool, optional
        If True, the function will compute the time/frame when pixel intensity falls below the threshold,
        rather than surpassing it. Defaults to False.
    interpolate : bool, optional
        If True, use linear interpolation to find the exact crossing point between frames. Defaults to False.
    min_duration : int, optional
        Minimum duration of the activation in frames. If set to 0, all activations are considered. Defaults to 0.
    normalize_time : bool, optional
        If True, the minimum activation time across all pixels will be subtracted from the activation times.
    set_nan_for_inactive : bool, optional
        If True, pixels that never reach the activation threshold or meet all the criteria will be set to NaN. If False, set to np.inf. Defaults to True.
    show : bool, optional
        If True, the resulting activation map will be displayed. Defaults to True.
    **kwargs : dict
        Additional arguments passed to :func:`show_activation_map`.

    Returns
    -------
    activation_map : ndarray
        2D image of activation times
    """
    _print(f"computing activation map with {threshold=}")
    if video.ndim != 3:
        msg = "video must be 3-dimensional"
        raise ValueError(msg)
    if "inverted" in kwargs:
        warnings.warn("`inverted` parameter is deprecated, use `falling_edge` instead", DeprecationWarning)
        falling_edge = kwargs.pop("inverted")
    
    _, height, width = video.shape
    amap = np.full((height, width), np.nan, dtype=np.float32)
    for x in range(width):
        for y in range(height):
            if np.isnan(video[:, y, x]).any():
                continue
            # Get the first activation for each pixel (if any)
            pixel_activations = find_activations(
                video[:, y, x],
                threshold=threshold,
                interpolate=interpolate,
                min_duration=min_duration,
                falling_edge=falling_edge,
                show=False
            )
            if len(pixel_activations) > 0:
                amap[y, x] = pixel_activations[0]
            else:
                amap[y, x] = np.nan if set_nan_for_inactive else np.inf
    
    if normalize_time:
        amap -= np.nanmin(amap)

    _print(f"minimum of activation_map: {np.nanmin(amap)}")
    _print(f"maximum of activation_map: {np.nanmax(amap)}")

    if show:
        kwargs["title"] = kwargs.get("title", "Activation Map")
        show_activation_map(amap, **kwargs)
    return amap
