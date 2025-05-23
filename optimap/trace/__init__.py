"""Functions for extracting time-series from videos."""

from ._compare import compare_traces
from ._core import (
    collage_positions,
    extract_traces,
    get_default_trace_window,
    set_default_trace_window,
    show_positions,
    show_trace,
    show_traces,
    show_positions_and_traces,
)
from ._detrend import detrend_timeseries
from ._interactive import select_positions, select_traces
from ._normalize import normalize
from ._points import positions_from_A_to_B, random_positions

__all__ = [
    "select_traces",
    "extract_traces",
    "select_positions",
    "compare_traces",

    "show_positions",
    "show_trace",
    "show_traces",
    "show_positions_and_traces",

    "normalize",

    "detrend_timeseries",

    "random_positions",
    "positions_from_A_to_B",

    "collage_positions",

    "set_default_trace_window",
    "get_default_trace_window",
]
