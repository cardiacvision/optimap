"""
Functions for loading, saving, and displaying images, and for creating masks.
"""
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print(
        "\n\nERROR: Unable to import opencv, optimap.image functions will be unavailable. Please install it, e.g. with `pip install opencv-python`.\n\n"
    )

if CV2_AVAILABLE:
    from ._core import show_image, load_image, load_mask, save_image
from ._mask import show_mask, detect_background_threshold, background_mask, foreground_mask, disc_mask, largest_mask_island

__all__ = [
    # _core.py
    "show_image",
    "load_image",
    "load_mask",
    "save_image",
    # _mask.py
    "show_mask",
    "detect_background_threshold",
    "background_mask",
    "foreground_mask",
    "disc_mask",
    "largest_mask_island",
]
