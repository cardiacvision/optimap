"""
Functions for extracting time-series from videos.
"""

from ._core import extract_traces, set_default_trace_window, get_default_trace_window, show_positions, show_trace, show_traces
from ._interactive import select_positions, select_traces
from ._compare import compare_traces
from ._points import random_positions, positions_from_A_to_B
from ._detrend import detrend_timeseries

__all__ = [
    # core
    'extract_traces',
    'set_default_trace_window',
    'get_default_trace_window',
    'show_positions',
    'show_trace',
    'show_traces',

    # interactive
    'select_positions',
    'select_traces',

    # compare
    'compare_traces',

    # points
    'random_positions',
    'positions_from_A_to_B',

    # detrend
    'detrend_timeseries',
]