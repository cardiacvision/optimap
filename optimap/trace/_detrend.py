import numpy as np
import seasonal


def detrend_timeseries(data, kind="spline", period=None, ptimes=2, retain_baseline=True, return_trend_data=False):
    """Detrends a time series using the `seasonal` package.

    Parameters
    ----------
    data : 1d array
        The time series to be detrended.
    kind : str
        One of ("mean", "median", "line", "spline", None)
        if mean, apply a period-based mean filter
        if median, apply a period-based median filter
        if line, fit a slope to median-filtered data.
        if spline, fit a piecewise cubic spline to the data
        if None, return zeros
    period : number
        seasonal periodicity, for filtering the trend.
        if None, will be estimated.
    ptimes : number
        multiple of period to use as smoothing window size
    retain_baseline : bool
        Retain baseline level of the time series by adding the mean of the trend back to the detrended time series.
    return_trend_data : bool
        Do not detrend, but return the trend data instead.

    Returns
    -------
    detrended or trend : ndarray
        The detrended time series (if `return_trend_data` is False), the calculated
        trend otherwise.
    """
    trend = seasonal.fit_trend(data, kind=kind, period=period, ptimes=ptimes)

    if return_trend_data:
        return trend
    elif retain_baseline:
        return data - trend + np.mean(trend)
    else:
        return data - trend
