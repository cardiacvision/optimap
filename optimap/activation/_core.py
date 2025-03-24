import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from ..image import show_image
from ..trace import show_traces


def show_activations(signal, activations, fps=None, ax=None, linecolor="red", linestyle="--", **kwargs):
    """Display a signal with vertical lines marking activation times.
    
    This function creates a visualization of a time series with clear markers at each
    activation point, making it easy to identify when activation events occur.
    
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
        _, ax = plt.subplots(figsize=(6.4, 4.8 / 2))
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

def show_activations_dvdt(signal, activations, fps=None, derivative=None, prominence=None, ax=None, linecolor="red", linestyle="--"):
    """Display a signal and its derivative with vertical lines marking activation times.
    
    This function creates plots showing both the original signal and its derivative with
    vertical lines indicating activation times, which is useful for visualizing the relationship
    between signal dynamics and detected activation events.
    
    Parameters
    ----------
    signal : array-like
        The 1D signal to display
    activations : array-like
        List of activation points (frame indices)
    fps : float, optional
        Frames per second for time conversion. If provided, x-axis will show time in seconds 
        instead of frame numbers.
    derivative : array-like, optional
        Pre-computed derivative of the signal. If None, only the signal will be plotted.
    prominence : float, optional
        Prominence threshold used for detection, shown as a horizontal line in the derivative plot.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure and axes is created.
    linecolor : str, optional
        Color of the vertical lines marking activation times, by default "red"
    linestyle : str, optional
        Line style of the vertical lines marking activation times, by default "--"
        
    Returns
    -------
    list of matplotlib.axes.Axes
        The axes containing the plots
    """
    if ax is None:
        if derivative is None:
            fig, ax = plt.subplots()
            axs = [ax]
        else:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    else:
        fig = ax.figure
        if derivative is None:
            axs = [ax]
        else:
            axs = []
            pos = ax.get_position()
            axs.append(fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2]))
            axs.append(fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2]))
            axs[1].sharex(axs[0])
            ax.axis("off")

    time = np.arange(len(signal))
    if fps is not None:
        time = time / fps
        xlabel = "Time [s]"
    else:
        fps = 1
        xlabel = "Frame"

    axs[0].plot(time, signal)
    for peak in activations:
        axs[0].axvline(peak / fps, color=linecolor, linestyle=linestyle, alpha=0.7)
    axs[0].set_ylabel("Intensity")
    axs[0].grid(True)

    if derivative is not None:
        axs[1].plot(time, derivative)
        for peak in activations:
            axs[1].axvline(peak / fps, color=linecolor, linestyle=linestyle, alpha=0.7)
        if prominence is not None:
            axs[1].axhline(prominence, color="g", linestyle=":", alpha=0.7, label=f"Prominence: {prominence:.4f}")                
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel("dV/dt")
        axs[1].grid(True)
        axs[1].legend()
    return axs

def find_activations_dvdt(signal, falling_edge=False, single_activation=False, window_size=5, prominence=None, min_distance=None, 
                         height=None, interpolate=False, fps=None, show=True, ax=None):
    """Find activation times of a 1D signal using the maximum derivative (dV/dt) method.

    Detects moments of activation by identifying where a signal's rate of change peaks.
    This approach is particularly effective for signals with sharp transitions, such as 
    action potentials or other biological activation events.

    For rising signals, the function finds where the positive derivative is maximized.
    For falling signals (set ``falling_edge=True``), it finds where the negative derivative is maximized.

    If ``single_activation`` is set to True, the timepoint of the maximum derivative is returned.
    Otherwise, the function detects peaks in the derivative signal which correspond to activation times.
    The parameters ``prominence``, ``height``, and ``min_distance``  are used to control the peak detection process, see :func:``scipy.signal.find_peaks`` for details.

    The function uses Savitzky-Golay filter to smooth the derivative calculation. If ``falling_edge`` is set to True,
    it detects the maximum negative derivative (for negative polarity signals). The ``window_size`` parameter controls
    length of the moving window applied in the Savitzky-Golay filter, see :func:`scipy.signal.savgol_filter` for details.

    
    Parameters
    ----------
    signal : array-like
        Input signal or array of signals.
    falling_edge : bool, optional
        If True, detect maximum negative derivative (for negative polarity signals). Default is False.
    single_activation : bool, optional
        If True, find only the timepoint of the maximum derivative. Default is False.
    window_size : int, optional
        Size of the window used for derivative calculation. Default is 5.
    prominence : float, optional
        Required prominence of peaks. If None, it will be calculated automatically
        based on the signal's derivative statistics. Default is None.
    min_distance : int, optional
        Minimum distance between detected peaks in frames. Default is None.
    height : float, optional
        Minimum height of peaks. Default is None.
    interpolate : bool, optional
        Whether to use interpolation for more precise activation time. Default is False.
    fps : float, optional
        Frames per second, used for time axis when plotting. Default is None.
    show : bool, optional
        Whether to show the detected activation times. Default is True.
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axes to plot on. If None, new figures are created. If a single axis is provided, 
        it will be split into two subplots.
    
    Returns
    -------
    activations : ndarray
        Detected activation times.
    """
    if signal.ndim > 1:
        raise NotImplementedError("signal input has to be 1D")

    # Use Savitzky-Golay filter for smoother derivative calculation
    try:
        derivative = savgol_filter(signal, window_size, 1, deriv=1)
    except ValueError:
        derivative = np.gradient(signal)

    if falling_edge:  # Negative polarity signals
        derivative = -derivative
    
    if single_activation:
        peak = np.argmax(derivative)
        peaks = np.array([peak])
    else:
        # Automatic prominence calculation based on derivative statistics
        if prominence is None:
            prominence = 3 * np.nanstd(derivative)
        peaks, properties = find_peaks(derivative,
                                    prominence=prominence,
                                    height=height,
                                    distance=min_distance)
    
    # Refine peak locations with interpolation if requested
    if interpolate and len(peaks) > 0:
        refined_peaks = []
        for peak in peaks:
            if peak > 0 and peak < len(derivative) - 1:
                # Parabolic interpolation using 3 points around peak
                y0, y1, y2 = derivative[peak-1:peak+2]
                if y0 < y1 and y2 < y1:  # Ensure it's a proper peak
                    peak_offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                    if -1 < peak_offset < 1:  # Ensure offset is reasonable
                        refined_peaks.append(peak + peak_offset)
                    else:
                        refined_peaks.append(peak)
                else:
                    refined_peaks.append(peak)
            else:
                refined_peaks.append(peak)
        peaks = np.array(refined_peaks)
    
    if show:
        show_activations_dvdt(signal, activations=peaks, derivative=derivative, fps=fps, ax=ax, prominence=prominence)
    return peaks

def find_activations_threshold(signal, threshold=0.5, interpolate=False, falling_edge=False, min_duration=8, fps=None, show=True, ax=None):
    """Find activation times by detecting threshold crossings in a 1D signal.
    
    Identifies the exact moments when a signal crosses a specified threshold value.
    This is useful for computing local activation times or detecting events in a time series.

    By default, it detects when the signal rises above the threshold, but can also
    detect when it falls below the threshold if ``falling_edge=True``.

    The function returns the **closest** frame index where the signal crosses the threshold. If ``interpolate`` is set to True, the function will return the exact crossing point using linear interpolation.

    Parameters
    ----------
        signal: np.ndarray
            The 1D signal.
        threshold: float, optional
            The threshold value. Default is 0.5.
        interpolate: bool, optional
            If True, use linear interpolation to find the exact crossing point. Default is False.
        falling_edge: bool, optional
            If True, find crossings from above to below the threshold. Default is False.
        min_duration: int, optional
            Minimum duration of the crossing in frames. If set to 0, all crossings are returned. Default is 8.
        fps: float, optional
            Frames per second for time conversion for plotting. Default is None.
        show: bool, optional
            Whether to show the activation times. Default is True.
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes is created. Default is None.

    Returns
    -------
        np.ndarray
    """
    if signal.ndim > 1:
        raise NotImplementedError("signal input has to be 1D")
    
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

def find_activations(signal, method="maximum_derivative", interpolate=False, falling_edge=False, show=True, fps=None, ax=None, **kwargs):
    """
    Computes activation times or detects activation events using different methods.

    A versatile function that identifies when activation events occur in time series data.
    
    Two different methods are available:

    #. **maximum_derivative**: Computes activation times based the peaks in the temporal derivative of the signal :math:`dV/dt`. See :func:`optimap.activation.find_activations_dvdt` for details.
    #. **threshold_crossing**: Computes activation times based on the signal crossing a specified threshold in desired direction. See :func:`optimap.activation.find_activations_threshold` for details.

    The function can handle 1D signals, 2D traces (T, N), or 3D videos (averaged over all spatial dimensions to compute 1D signal).

    If ``falling_edge`` is set to True, it expects *negative activation* (i.e., the signal falls below the threshold).

    By default, the function returns the _closest_ frame index to the condition. If ``interpolate`` is set to True, the function will return a fractional frame using interpolation.

    Parameters
    ----------
        signal: np.ndarray
            The 1D signal.
        method: str
            The method to use for finding activations. Options are "maximum_derivative" or "threshold".
        falling_edge: bool
            If True, find activations which go negative (falling edge).
        interpolate: bool
            If True, use linear interpolation to find the exact crossing point.
        threshold: float
            If ``method='threshold_crossing'``, the threshold value.
        min_duration: int
            If ``method='threshold_crossing'``, the minimum duration of the crossing in frames. If set to 0, all crossings are returned.
        prominence: float
            If ``method='maximum_derivative'``, the required prominence of peaks. If None, it will be calculated automatically based on the signal's derivative statistics.
        height: float
            If ``method='maximum_derivative'``, the minimum height of peaks. Default is None.
        min_distance: int
            If ``method='maximum_derivative'``, the minimum distance between detected peaks in frames. Default is 10.
        window_size: int
            If ``method='maximum_derivative'``, the size of the window used for derivative calculation. Default is 5.
        show: bool
            If True, the resulting activation times will be displayed.
        fps: float
            Frames per second for time conversion for plotting.
        ax: matplotlib.axes.Axes
            Axes on which to plot. If None, a new figure and axes is created.
        **kwargs: dict
            Additional arguments passed to the underlying activation detection functions.

    Returns
    -------
    activations : ndarray
        Detected activation times.
    """
    if signal.ndim == 3:
        # mean signal over the spatial dimensions
        signal = np.nanmean(signal, axis=(1, 2))
    elif signal.ndim == 2:
        # signal is a trace (T, N)
        return [
            find_activations(signal[:, i], method=method, interpolate=interpolate, falling_edge=falling_edge, fps=fps, show=False, **kwargs)
                             for i in range(signal.shape[1])
        ]
    elif signal.ndim != 1:
        raise ValueError("Error: signal is not a video or trace.")
    
    if method == "maximum_derivative":
        return find_activations_dvdt(signal, interpolate=interpolate, falling_edge=falling_edge, fps=fps, show=show, ax=ax, **kwargs)
    elif method == "threshold_crossing":
        return find_activations_threshold(signal, interpolate=interpolate, falling_edge=falling_edge, fps=fps, show=show, ax=ax, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'")


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

    Creates a color-coded visualization of activation timing across a 2D spatial region.
    Optional contour lines can be added to highlight regions that activate at the same time
    (isochrones).

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
        Title for the colorbar, by default "Activation Time [ms]" if ``fps`` is provided, otherwise "Activation Time [frames]"
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes is created, by default None
    contour_fontsize : int or float, optional
        Font size for contour labels, by default None (auto-determined)
    contour_fmt : str, optional
        Format string for contour labels e.g. ``' %1.0f ms '``, by default None. See :func:`matplotlib.pyplot.clabel` for details.
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
    if activation_map.ndim != 2:
        raise ValueError("activation_map must be 2D array")
    
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
                           method="maximum_derivative",
                           falling_edge=False,
                           interpolate=False,
                           threshold=0.5,
                           normalize_time=True,
                           set_nan_for_inactive=True,
                           show=True,
                           **kwargs):
    """Computes an activation map (or isochrone map) of local activation times from a video.

    The activation map is a 2D image where each pixel represents the time (in terms of frame index) at which the pixel is considered activated.

    The activation time is determined by  :func:`find_activations` based on the specified method, which can be either "threshold_crossing" or "maximum_derivative".

    * ``threshold_crossing``: The activation time is the first frame where the pixel intensity surpasses a specified threshold.
    * ``maximum_derivative``: The activation time is the where the maximum of the temporal derivative of the pixel intensity occurs.

    .. note::
        For the ``threshold_crossing`` method the video should be normalized to the range [0, 1]. This is not required for the ``maximum_derivative`` method.

    For negative polarity signals (i.e. inverted action potentials), set ``falling_edge=True``.
    The activation map is return in terms of discrete frame indices, if ``interpolate=True`` fractions of frames are returned. If ``normalize_time=True`` (default), the minimum activation time across all pixels will be subtracted from the activation times.
    
    See :func:`find_activations` for further details and :func:`show_activation_map` for plotting.

    Parameters
    ----------
    video : np.ndarray
        A 3D array representing the video, with dimensions {t (time or frames), x (width), y (height)}.
    method : str, optional
        Method to compute activation times. Options are "threshold_crossing" or "maximum_derivative". Defaults to "maximum_derivative".
    falling_edge : bool, optional
        If True, the function will compute the time/frame when pixel intensity falls below the threshold,
        rather than surpassing it. Defaults to False.
    interpolate : bool, optional
        If True, use linear interpolation to find the exact crossing point between frames. Defaults to False.
    threshold : float, optional
        Intensity threshold for ``threshold_crossing`` method at which a pixel is considered activated. Defaults to 0.5.
    normalize_time : bool, optional
        If True, the minimum activation time across all pixels will be subtracted from the activation times. Defaults to True.
    set_nan_for_inactive : bool, optional
        If True, pixels that never reach the activation threshold or meet all the criteria will be set to NaN. 
        If False, set to np.inf. Defaults to True.
    show : bool, optional
        If True, the resulting activation map will be displayed. Defaults to True.
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_activation_map`.

    Returns
    -------
    activation_map : ndarray
        2D image of activation times
    """
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
            activation_kwargs = {}
            if method == "threshold_crossing":
                activation_kwargs = {"threshold": threshold}
            elif method == "maximum_derivative":
                activation_kwargs = {"single_activation": True}
            pixel_activations = find_activations(
                video[:, y, x],
                method=method,
                interpolate=interpolate,
                falling_edge=falling_edge,
                show=False,
                **activation_kwargs,
            )
            if len(pixel_activations) > 0:
                amap[y, x] = pixel_activations[0]
            else:
                amap[y, x] = np.nan if set_nan_for_inactive else np.inf
    
    if normalize_time:
        amap -= np.nanmin(amap)

    if show:
        kwargs["title"] = kwargs.get("title", "Activation Map")
        show_activation_map(amap, **kwargs)
    return amap
