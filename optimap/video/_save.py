import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import skimage.io as sio
import tifffile
from scipy.io import savemat


def save_image_sequence(video: np.ndarray,
                        filepattern: str = "frame_{:03d}",
                        directory: Union[str, Path] = None,
                        suffix=".png",
                        **kwargs):
    """Save a video as a sequence of images.

    This function will create a directory if it does not exist, and save each frame of the video as an image.
    The images are named according to the provided file pattern and suffix.

    Example
    -------

    .. code-block:: python
    
            import numpy as np
            import optimap as om
    
            video = np.random.rand(100, 100, 100)
            om.save_image_sequence(video, filepattern="frame_{:03d}", directory="my_folder", suffix=".png")
        
    Will create a folder ``my_folder`` and save the images as ``my_folder/frame_000.png``, ``my_folder/frame_001.png``, etc.

    Parameters
    ----------
    video : np.ndarray or list of np.ndarray
        The video to save as a sequence of images
    filepattern : str
        The pattern to use for the filenames. The pattern should include a placeholder for the frame number, e.g., 'frame_{:03d}'. Default is 'frame_{:03d}'.
    directory : str or pathlib.Path, optional
        Directory to save to, by default None
    **kwargs : dict
        Additional arguments to pass to :func:`skimage.io.imsave` or :func:`tifffile.imwrite` (for ``.tiff`` files)
    """
    if directory is not None:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)
    else:
        directory = Path.cwd()

    for i, frame in enumerate(video):
        fn = filepattern.format(i) + suffix
        fn = directory / fn
        if suffix.lower() in [".tif", ".tiff"]:
            save_tiff(frame, fn, **kwargs)
        else:
            sio.imsave(fn, frame, **kwargs)


def save_tiff(video, filename, photometric="minisblack", **kwargs):
    """Save a video as a monochromatic TIFF stack.

    Parameters
    ----------
    video : np.ndarray
        Video to save
    filename : str or pathlib.Path
        Filename to save to
    photometric : str, optional
        Photometric interpretation, by default ``'minisblack'``
    **kwargs : dict
        Additional arguments to pass to :func:`tifffile.imwrite`
    """
    filename = Path(filename)
    if filename.suffix.lower() not in [".tif, .tiff"]:
        filename = filename.with_suffix(".tiff")
    print(f"saving video to tiff stack {filename}")
    tifffile.imwrite(filename, video, photometric=photometric, **kwargs)


def save_tiff_folder(video, filepattern="frame_{:03d}", directory=None):
    """Save a video as a folder of TIFF images. See :func:`save_image_sequence` for details.

    Parameters
    ----------
    video : np.ndarray
        Video to save
    filepattern : str
        File pattern of the images, by default ``'frame_{:03d}'``
    directory : str or pathlib.Path, optional
        Directory to save to, by default current working directory
    """
    save_image_sequence(video, filepattern=filepattern, directory=directory, suffix=".tiff")


def save_matlab(array, filename, fieldname="video", appendmat=True):
    """Save an array to a MATLAB ``.mat`` file.

    Parameters
    ----------
    array : np.ndarray
        Array to save
    filename : str or pathlib.Path
        Filename to save to
    fieldname : str, optional
        Name of the field to save the array to, by default ``'video'``
    appendmat : bool, optional
        Whether to append to an existing file, by default ``True``
    """
    savemat(filename, {fieldname: array}, appendmat=appendmat)


def save_video(filename, video, **kwargs):
    """Save a video to a file. Supported file formats are ``.npy``, ``.tif``/``.tiff``, and ``.mat``.

    See :func:`save_tiff_folder` for saving a video as a folder of ``.tiff`` images.

    See :func:`export_video` for exporting a video to a movie file (e.g. ``.mp4``).

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename to save to
    video : np.ndarray
        Video to save
    **kwargs : dict
        Additional arguments to pass to the respective save function
    """
    if isinstance(video, (str, os.PathLike)):
        filename, video = video, filename
        warnings.warn("The order of arguments for optimap.save_video() has changed. "
                      "Please use save_video(filename, video) instead of save_video(video, filename).",
                      DeprecationWarning)

    filename = Path(filename)
    suffix = filename.suffix.lower()
    if not suffix:
        msg = f"File '{filename}' has no file extension, unable to save video"
        raise RuntimeError(msg)

    if suffix in [".tif", ".tiff"]:
        save_tiff(video, filename)
    elif suffix in [".npy"]:
        np.save(filename, video, **kwargs)
        print(f"saved video as numpy file to {filename}")
    elif suffix in [".mat"]:
        save_matlab(video, filename, **kwargs)
    else:
        msg = f"Unrecognized file extension {suffix} (for file '{filename}')"
        raise RuntimeError(msg)
