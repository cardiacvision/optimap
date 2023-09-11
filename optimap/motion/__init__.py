"""
Motion estimation and compensation functions for optical mapping data.

The function :py:func:`motion_compensate` is main top-level function which combines all the steps of the motion compensation pipeline.

See :footcite:t:`Christoph2018a` and :footcite:t:`Lebert2022` for details.

.. footbibliography::
"""

__all__ = [
    "FlowEstimator",
    "warp_video",
    "warp_image",
    "contrast_enhancement",
    "smooth_displacements",
    "estimate_displacements",
    "estimate_reverse_displacements",
    "motion_compensate",
    "play_displacements",
    "play_displacements_points",
]

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print(
        "\n\nERROR: Unable to import opencv, optimap.motion functions will be unavailable. Please install it, e.g. with `pip install opencv-python`.\n\n"
    )

if CV2_AVAILABLE:
    from ._flowestimator import FlowEstimator
    from ._warping import warp_video, warp_image
    from ._core import contrast_enhancement, smooth_displacements, estimate_displacements, estimate_reverse_displacements, motion_compensate
from ._plot import play_displacements, play_displacements_points
