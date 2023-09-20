"""
Functions for loading, viewing, filtering, saving and exporting videos.
"""

from ._edit import (
    rotate_left,
    rotate_right,
    flip_up_down,
    flip_left_right,
    resize,
    crop,
    pad,
)

from ._filters import (
    normalize,
    normalize_pixelwise,
    normalize_pixelwise_slidingwindow,
    smooth_spatiotemporal,
    evolve_jitter_filter,
    temporal_difference,
)
from ._play import play, play2, playn, play_with_overlay
from ._export import (
    export_video,
    export_video_with_overlay,
    iter_alpha_blend_videos,
    alpha_blend_videos,
    smoothstep,
    set_ffmpeg_defaults,
    set_default_ffmpeg_encoder,
    get_default_ffmpeg_encoder,
)
from ._load import (
    load_video,
    load_metadata,
)
from ._importers import (
    MultiRecorderImporter,
    MiCAM05_Importer,
    MiCAM_ULTIMA_Importer,
)
from ._save import (
    save_matlab,
    save_tiff,
    save_tiff_folder,
    save_video,
    save_image_sequence,
)

__all__ = [
    "load_video",
    "load_metadata",

    "save_video",
    "save_image_sequence",
    "save_matlab",
    "save_tiff",
    "save_tiff_folder",

    "export_video",
    "export_video_with_overlay",

    "play",
    "play2",
    "playn",
    "play_with_overlay",

    "rotate_left",
    "rotate_right",
    "flip_up_down",
    "flip_left_right",
    "resize",
    "crop",
    "pad",

    "normalize",
    "normalize_pixelwise",
    "normalize_pixelwise_slidingwindow",
    "smooth_spatiotemporal",
    "temporal_difference",

    "evolve_jitter_filter",

    "alpha_blend_videos",
    "iter_alpha_blend_videos",
    "smoothstep",
    "set_ffmpeg_defaults",
    "set_default_ffmpeg_encoder",
    "get_default_ffmpeg_encoder",

    "MultiRecorderImporter",
    "MiCAM05_Importer",
    "MiCAM_ULTIMA_Importer",
]
