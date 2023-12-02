"""Motion estimation and compensation functions for optical mapping data.

The function :py:func:`motion_compensate` is main top-level function which combines all steps
of the motion compensation pipeline for optical mapping data.

See :footcite:t:`Christoph2018a` and :footcite:t:`Lebert2022` for details.

.. footbibliography::
"""

__all__ = [
    "motion_compensate",
    "reverse_motion_compensate",
    "estimate_displacements",
    "estimate_reverse_displacements",
    "warp_video",
    "warp_image",
    "contrast_enhancement",
    "smooth_displacements",
    "play_displacements",
    "play_displacements_points",
    "FlowEstimator",
]

from ._core import (
    contrast_enhancement,
    estimate_displacements,
    estimate_reverse_displacements,
    motion_compensate,
    reverse_motion_compensate,
    smooth_displacements,
)
from ._flowestimator import FlowEstimator
from ._plot import play_displacements, play_displacements_points
from ._warping import warp_image, warp_video
