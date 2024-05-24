"""Functions for loading, saving, and displaying images, and for creating masks."""

from ._core import collage, load_image, load_mask, save_image, export_image, save_mask, show_image, smooth_gaussian
from ._edit import normalize, crop, flip_left_right, flip_up_down, pad, resize, rotate_left, rotate_right
from ._mask import (
    background_mask,
    close_mask,
    detect_background_threshold,
    dilate_mask,
    disc_mask,
    erode_mask,
    fill_mask,
    foreground_mask,
    interactive_mask,
    invert_mask,
    largest_mask_component,
    open_mask,
    show_mask,
)

__all__ = [
    "show_image",
    "show_mask",
    "save_image",
    "export_image",
    "save_mask",
    "load_image",
    "load_mask",

    "resize",
    "rotate_left",
    "rotate_right",
    "flip_left_right",
    "flip_up_down",
    "crop",
    "pad",
    "normalize",
    "collage",
    "smooth_gaussian",

    "interactive_mask",
    "foreground_mask",
    "background_mask",
    "detect_background_threshold",
    "invert_mask",

    "erode_mask",
    "dilate_mask",
    "largest_mask_component",
    "fill_mask",

    "disc_mask",
    "open_mask",
    "close_mask",
]
