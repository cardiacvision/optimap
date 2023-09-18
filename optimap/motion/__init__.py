"""
Motion estimation and compensation functions for optical mapping data.

The function :py:func:`motion_compensate` is main top-level function which combines all the steps of the motion compensation pipeline.

See :footcite:t:`Christoph2018a` and :footcite:t:`Lebert2022` for details.

.. footbibliography::
"""

__all__ = [
    "motion_compensate",
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

from ._flowestimator import FlowEstimator
from ._warping import warp_video, warp_image
from ._core import contrast_enhancement, smooth_displacements, estimate_displacements, estimate_reverse_displacements, motion_compensate
from ._plot import play_displacements, play_displacements_points
