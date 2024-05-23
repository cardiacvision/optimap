"""Functions for loading, viewing, filtering, saving and exporting videos."""

from ..image import normalize
from ._edit import (
    crop,
    flip_left_right,
    flip_up_down,
    pad,
    resize,
    rotate_left,
    rotate_right,
)
from ._export import (
    alpha_blend_videos,
    export_video,
    export_videos,
    export_video_collage,
    export_video_with_overlay,
    get_default_ffmpeg_encoder,
    iter_alpha_blend_videos,
    set_default_ffmpeg_encoder,
    set_ffmpeg_defaults,
    smoothstep,
)
from ._filters import (
    evolve_jitter_filter,
    normalize_pixelwise,
    normalize_pixelwise_slidingwindow,
    smooth_spatiotemporal,
    temporal_difference,
)
from ._importers import (
    MiCAM05_Importer,
    MiCAM_ULTIMA_Importer,
    MultiRecorderImporter,
)
from ._load import (
    load_metadata,
    load_video,
    load_image_folder
)
from ._play import (
    # deprecated
    play,
    play2,
    play_with_overlay,
    playn,
    # new names
    show_video,
    show_video_pair,
    show_videos,
    show_video_overlay
)
from ._player import InteractivePlayer
from ._save import (
    save_image_sequence,
    save_matlab,
    save_tiff,
    save_tiff_folder,
    save_video,
)

__all__ = [
    "load_video",
    "load_metadata",
    "load_image_folder",

    "save_video",
    "save_image_sequence",
    "save_matlab",
    "save_tiff",
    "save_tiff_folder",

    "export_video",
    "export_videos",
    "export_video_with_overlay",

    # deprecated
    "play",
    "play2",
    "playn",
    "play_with_overlay",
    "export_video_collage",
    
    # new names
    "show_videos",
    "show_video",
    "show_video_pair",
    "show_video_overlay",

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

    "InteractivePlayer",

    "MultiRecorderImporter",
    "MiCAM05_Importer",
    "MiCAM_ULTIMA_Importer",
]
