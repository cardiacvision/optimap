import re
from pathlib import Path

import numpy as np
import skimage.io as sio
from tifffile import imread as tifffile_imread, memmap as tifffile_memmap
from scipy.io import loadmat

from ..utils import _print
from ._importers import MultiRecorderImporter, MiCAM05_Importer, MiCAM_ULTIMA_Importer


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
    """
    Loads a video from a .tiff stack.
    """
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
    """
    Loads a sequences of images from a folder as a video.

    Supported extensions (case-insensitive):

    * .tif and .tiff
    * .png


    Parameters
    ----------
    path : str or pathlib.Path
        Path to folder containing .tiff images
    prefix : str, optional
        Prefix of the files to load
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
        raise ValueError(f"'{path}' is not a directory")

    files = []
    for extension in [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG"]:
        files.extend(path.glob(f"{prefix}*{extension}"))
        if len(files) > 0:
            break
    if not files:
        raise ValueError(f"No .tif, .tiff or .png files found in folder '{path}'")

    files = sorted(files, key=_natural_sort_path_key)
    _print(f"found {len(files)} files to load in '{path}'")
    files = files[start_frame:end_frame:step]
    _print(
        f"loading '{files[0].name}' as first frame, and '{files[-1].name}' as last frame"
    )
    video = [sio.imread(file) for file in files]
    return np.array(video)


def load_MATLAB(filename, fieldname=None, start_frame=0, end_frame=None, step=1):
    """
    Loads a video from a MATLAB .mat file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to .mat file
    fieldname : str, optional
        Name of the field to load, by default None. If None, loads the first field.
    """

    data = loadmat(filename)
    if fieldname is None:
        fields = [key for key in data.keys() if not key.startswith("__")]
        if len(fields) == 0:
            raise ValueError(f"No fields found in file '{filename}'")
        elif len(fields) > 1:
            print(f"WARNING: Multiple fields found in file '{filename}', loading first field '{fields[0]}'"
            )
        fieldname = fields[0]
    video = data[fieldname]
    return video[start_frame:end_frame:step]


def load_MultiRecorder(filepath, start_frame=0, frames=None, step=1, use_mmap=False):
    dat = MultiRecorderImporter(filepath)
    video = dat.load_video(start_frame=start_frame, frames=frames, step=step, use_mmap=use_mmap)
    return video

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
    dat = MiCAM_ULTIMA_Importer(filename)
    video = dat.load_video(start_frame=start_frame, frames=frames, step=step)
    return video

def load_SciMedia_MiCAMULTIMA_metadata(filename):
    dat = MiCAM_ULTIMA_Importer(filename)
    return dat.get_metadata()

def load_video(path, start_frame=0, frames=None, step=1, use_mmap=False):
    """
    Loads a video from a file or folder, automatically detecting the file type.

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
    
    For some file types read-only memory mapping is used when ``use_mmap=True``. This is useful for large videos, as it does not load the entire video into memory, only loading the frames when they are accessed. However, it is not supported for all file types, and it is not possible to write to the video array. Supported file types for memory mapping:

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
        raise ValueError(f"File or folder '{path}' does not exist")
    elif path.is_dir():
        return load_image_folder(path, start_frame=start_frame, end_frame=end_frame, step=step)
    elif suffix in [".tif", ".tiff"]:
        return load_tiff(path, start_frame=start_frame, end_frame=end_frame, step=step, use_mmap=use_mmap)
    elif suffix in [".mat"]:
        return load_MATLAB(path, start_frame=start_frame, end_frame=end_frame, step=step)
    elif suffix in [".gsd", ".gsh"]:
        return load_SciMedia_MiCAM05(path, start_frame=start_frame, frames=frames, step=step)
    elif suffix in [".rsh", ".rsm", ".rsd"]:
        return load_SciMedia_MiCAMULTIMA(path, start_frame=start_frame, frames=frames, step=step)
    elif suffix in [".dat"]:
        return load_MultiRecorder(path, start_frame=start_frame, frames=frames, step=step, use_mmap=use_mmap)
    elif suffix in [".npy"]:
        return load_numpy(path, start_frame=start_frame, end_frame=end_frame, step=step, use_mmap=use_mmap)
    else:
        raise ValueError(
            f"Unable to find videoloader for file extension {suffix} (for file '{path}')"
        )

def load_metadata(filename):
    """
    Loads metadata information from a recording, automatically detecting the file type.

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
        raise ValueError(f"'{filename}' is a directory, not a file")
    elif not filename.exists():
        raise ValueError(f"File '{filename}' does not exist")

    suffix = filename.suffix.lower()
    if suffix in [".gsd", ".gsh"]:
        return load_SciMedia_MiCAM05_metadata(filename)
    elif suffix in [".rsh", ".rsm", ".rsd"]:
        return load_SciMedia_MiCAMULTIMA_metadata(filename)
    elif suffix in [".dat"]:
        return load_MultiRecorder_metadata(filename)
    else:
        raise ValueError(
            f"Unable to find videoloader for file extension {suffix} (for file '{filename}')"
        )