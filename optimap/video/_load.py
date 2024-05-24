import re
import warnings
from pathlib import Path

import numpy as np
import skimage.io as sio
import skvideo.io
from scipy.io import loadmat
from tifffile import imread as tifffile_imread
from tifffile import memmap as tifffile_memmap

from ..utils import _print
from ._importers import MiCAM05_Importer, MiCAM_ULTIMA_Importer, MultiRecorderImporter
from ._export import _fix_ffmpeg_location


def _natural_sort_path_key(path: Path, _nsre=re.compile("([0-9]+)")):
    return [
        int(text) if text.isdigit() else text.lower() for text in _nsre.split(path.name)
    ]


def load_numpy(filename, start_frame=0, end_frame=None, step=1, use_mmap=False):
    video = np.load(filename, mmap_mode="r")
    video = video[start_frame:end_frame:step]
    if not use_mmap:
        video = video.copy()
    return video


def load_tiff(filename, start_frame=0, end_frame=None, step=1, use_mmap=False):
    """Loads a video from a .tiff stack."""
    if use_mmap:
        _print(f"loading video: {filename} with memory mapping ...")
        video = tifffile_memmap(filename, mode="r")
        video = video[start_frame:end_frame:step]
    else:
        _print(f"loading video: {filename} ...")
        video = tifffile_imread(filename)
        video = video[start_frame:end_frame:step]
    _print(
        f"finished loading video '{filename}' with shape {video.shape} and dtype {video.dtype}"
    )
    return video

def load_image_folder(path, prefix="", start_frame=0, end_frame=None, step=1):
    """Loads a sequences of images from a folder as a video.

    The filenames are sorted in natural order (i.e 'frame_2.png' comes before 'frame_10.png') and loaded in that order. The `prefix` parameter can be used to filter the files to load.

    The `start_frame`, `end_frame` and `step` parameters can be used to load only a subset of the images. Note that they refer to the index of the images in the sorted list of files.

    Supported image extensions (case-insensitive):

    * .tif and .tiff
    * .png


    Parameters
    ----------
    path : str or pathlib.Path
        Path to folder containing the images
    prefix : str, optional
        Prefix of the image files to load, by default "".
    start_frame : int, optional
        Index of the starting frame (0-indexed). Defaults to 0.
    end_frame : int or None, optional
        Index of the ending frame (non-inclusive). If None, loads all frames till the end.
    step : int, optional
        Steps between frames. If greater than 1, it will skip frames. Defaults to 1.

    Returns
    -------
    np.ndarray
        3D numpy array containing the loaded images
    """
    _print(f"loading video from series of images files in folder '{path}'")
    path = Path(path)
    if not path.is_dir():
        msg = f"'{path}' is not a directory"
        raise ValueError(msg)

    files = []
    for extension in [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG"]:
        files.extend(path.glob(f"{prefix}*{extension}"))
        if len(files) > 0:
            break
    if not files:
        msg = f"No .tif, .tiff or .png files found in folder '{path}'"
        raise ValueError(msg)

    files = sorted(files, key=_natural_sort_path_key)
    _print(f"found {len(files)} files to load in '{path}'")
    files = files[start_frame:end_frame:step]
    _print(f"loading '{files[0].name}' as first frame, and '{files[-1].name}' as last frame")
    video = [sio.imread(file) for file in files]
    return np.array(video)


def load_MATLAB(filename, fieldname=None, start_frame=0, end_frame=None, step=1):
    """Loads a video from a MATLAB .mat file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to .mat file
    fieldname : str, optional
        Name of the field to load, by default None. If None, loads the first field.
    start_frame : int, optional
        Index of the starting frame (0-indexed). Defaults to 0.
    end_frame : int or None, optional
        Index of the ending frame (non-inclusive). If None, loads all frames till the end.
    step : int, optional
        Steps between frames. If greater than 1, it will skip frames. Defaults to 1.
    """
    data = loadmat(filename)
    if fieldname is None:
        fields = [key for key in data.keys() if not key.startswith("__")]
        if len(fields) == 0:
            msg = f"No fields found in file '{filename}'"
            raise ValueError(msg)
        elif len(fields) > 1:
            warnings.warn(f"Multiple fields found in file '{filename}', loading first field '{fields[0]}'", UserWarning)
        fieldname = fields[0]
    video = data[fieldname]
    return video[start_frame:end_frame:step]


def load_MultiRecorder(filepath, start_frame=0, frames=None, step=1, use_mmap=False):
    dat = MultiRecorderImporter(filepath)
    return dat.load_video(start_frame=start_frame, frames=frames, step=step, use_mmap=use_mmap)

def load_MultiRecorder_metadata(filepath):
    dat = MultiRecorderImporter(filepath)
    return dat.get_metadata()

def load_SciMedia_MiCAM05(filename, start_frame=0, frames=None, step=1):
    _print(f"loading video: {filename} ...")

    dat = MiCAM05_Importer(filename)
    video = dat.load_video(start_frame=start_frame, frames=frames, step=step)

    _print(f"finished loading video '{filename}' with shape {video.shape} and dtype {video.dtype}.")
    _print(f"Experiment from {dat._meta['date']} acquired at {dat._meta['framerate']} fps.")
    return video

def load_SciMedia_MiCAM05_metadata(filename):
    dat = MiCAM05_Importer(filename)
    return dat.get_metadata()

def load_SciMedia_MiCAMULTIMA(filename, start_frame=0, frames=None, step=1):
    importer = MiCAM_ULTIMA_Importer(filename)
    return importer.load_video(start_frame=start_frame, frames=frames, step=step)

def load_SciMedia_MiCAMULTIMA_metadata(filename):
    importer = MiCAM_ULTIMA_Importer(filename)
    return importer.get_metadata()

def load_encoded_video(filename, start_frame=0, end_frame=None, step=1, as_grey=True):
    _fix_ffmpeg_location()
    if end_frame is None:
        num_frames = 0
    else:
        num_frames = end_frame
    try:
        data = skvideo.io.vread(
            str(filename),
            num_frames=num_frames,
            as_grey=as_grey
        )
    except AttributeError as e:
        raise AttributeError("Error during video load. Make sure you've updated scikit-video using 'pip install --upgrade --no-deps --force-reinstall git+https://github.com/scikit-video/scikit-video.git'") from e
    data = data[start_frame:end_frame:step]
    if as_grey:
        data = data[..., 0]
    return data

def load_video(path, start_frame=0, frames=None, step=1, use_mmap=False, **kwargs):
    """Loads a video from a file or folder, automatically detecting the file type.

    If ``path`` is a folder, it will load a video from a series of images in the folder.

    Supported file types:

    - .tif, .tiff (TIFF stack)
    - .mat (MATLAB), loads the first field in the file
    - .dat (MultiRecorder)
    - .gsd, .gsh (SciMedia MiCAM 05)
    - .rsh, .rsm, .rsd (SciMedia MiCAM ULTIMA)
    - .npy (numpy array)

    Supported file types when loading a folder:

    - .tif, .tiff images
    - .png images

    For some file types read-only memory mapping is used when ``use_mmap=True``. This is useful for large videos,
    as it does not load the entire video into memory, only loading the frames when they are accessed. However, it
    is not supported for all file types, and it is not possible to write to the video array. Supported file types
    for memory mapping:

    - .tif, .tiff (TIFF stack)
    - .dat (MultiRecorder)
    - .npy (numpy array)

    Parameters
    ----------
    path : str or pathlib.Path
        Path to video file, or folder containing images
    start_frame : int, optional
        Index of the starting frame (0-indexed). Defaults to 0.
    frames : int or None, optional
        Number of frames to load. If None, loads all frames till the end.
    step : int, optional
        Steps between frames. If greater than 1, it will skip frames. Defaults to 1.
    use_mmap : bool, optional
        If True, uses memory mapping to load the video in read-only mode, defaults to False.
        Only supported for some file types.
    **kwargs : dict
        Additional arguments to pass to the video loader function

    Returns
    -------
    video : {t, x, y} ndarray
        Video array
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if step > 1 and frames is not None:
        frames *= step

    if frames is not None:
        end_frame = start_frame + frames
    else:
        end_frame = None

    if not path.exists():
        msg = f"File or folder '{path}' does not exist"
        raise FileNotFoundError(msg)
    elif path.is_dir():
        return load_image_folder(path, start_frame=start_frame, end_frame=end_frame, step=step, **kwargs)
    elif suffix in [".tif", ".tiff"]:
        return load_tiff(path, start_frame=start_frame, end_frame=end_frame, step=step, use_mmap=use_mmap, **kwargs)
    elif suffix in [".mat"]:
        return load_MATLAB(path, start_frame=start_frame, end_frame=end_frame, step=step, **kwargs)
    elif suffix in [".gsd", ".gsh"]:
        return load_SciMedia_MiCAM05(path, start_frame=start_frame, frames=frames, step=step, **kwargs)
    elif suffix in [".rsh", ".rsm", ".rsd"]:
        return load_SciMedia_MiCAMULTIMA(path, start_frame=start_frame, frames=frames, step=step, **kwargs)
    elif suffix in [".dat"]:
        return load_MultiRecorder(path, start_frame=start_frame, frames=frames, step=step, use_mmap=use_mmap, **kwargs)
    elif suffix in [".npy"]:
        return load_numpy(path, start_frame=start_frame, end_frame=end_frame, step=step, use_mmap=use_mmap, **kwargs)
    elif suffix in [".mp4", ".avi", ".webm", ".mov", ".mkv", ".gif", ".wmv", ".m4v"]:
        return load_encoded_video(path, start_frame=start_frame, end_frame=end_frame, step=step, **kwargs)
    else:
        msg = f"Unable to find videoloader for file extension {suffix} (for file '{path}')"
        raise ValueError(msg)

def load_metadata(filename):
    """Loads metadata information from a recording, automatically detecting the file type.

    Supported file types:

    - .gsd, .gsh (SciMedia MiCAM 05)
    - .rsh, .rsm, .rsd (SciMedia MiCAM ULTIMA)
    - .dat (MultiRecorder)

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to video file

    Returns
    -------
    dict
        Dictionary containing the metadata
    """
    filename = Path(filename)
    if filename.is_dir():
        msg = f"'{filename}' is a directory, not a file"
        raise ValueError(msg)
    elif not filename.exists():
        msg = f"File '{filename}' does not exist"
        raise ValueError(msg)

    suffix = filename.suffix.lower()
    if suffix in [".gsd", ".gsh"]:
        return load_SciMedia_MiCAM05_metadata(filename)
    elif suffix in [".rsh", ".rsm", ".rsd"]:
        return load_SciMedia_MiCAMULTIMA_metadata(filename)
    elif suffix in [".dat"]:
        return load_MultiRecorder_metadata(filename)
    else:
        msg = f"Unable to find videoloader for file extension {suffix} (for file '{filename}')"
        raise ValueError(msg)
