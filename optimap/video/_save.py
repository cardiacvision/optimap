from pathlib import Path

import numpy as np
import tifffile

import skimage.io as sio


def save_image_sequence(video: np.ndarray, filepattern: str = 'frame_{:03d}', directory: str = None, suffix='.png', **kwargs):
    """
    Save a video as a sequence of images.
    
    Parameters
    ----------
    video : np.ndarray
        Video to save
    filepattern : str
        File pattern to save to, by default ``'frame_{:03d}'``
    suffix : str, optional
        Extension to use, by default ``'.png'``
    directory : str or pathlib.Path, optional
        Directory to save to, by default None
    **kwargs : dict
        Additional arguments to pass to :func:`skimage.io.imsave` or :func:`tifffile.imwrite` (for ``.tiff`` files)
    """

    for i, frame in enumerate(video):
        fn = filepattern.format(i) + suffix
        if directory is not None:
            fn = Path(directory) / fn
        if suffix.lower() in ['.tif', '.tiff']:
            save_tiff(frame, fn, **kwargs)
        else:
            sio.imsave(fn, frame, **kwargs)


def save_tiff(video, filename, photometric='minisblack', **kwargs):
    """
    Save a video as a monochromatic TIFF stack.

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
    if filename.suffix.lower() not in ['.tif, .tiff']:
        filename = filename.with_suffix('.tiff')
    print(f"saving video to tiff stack {filename}")
    tifffile.imwrite(filename, video, photometric=photometric, **kwargs)


def save_tiff_folder(video, filepattern, directory=None):
    save_image_sequence(video, filepattern, directory=directory, suffix='.tiff')


def save_matlab(array, filename, fieldname='video', appendmat=True):
    """
    Save an array to a MATLAB ``.mat`` file.

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
    sio.savemat(filename, {fieldname: array}, appendmat=appendmat)


def save_video(video, filename, **kwargs):
    """
    Save a video to a file. Supported file formats are ``.npy``, ``.tif``/``.tiff``, and ``.mat``.
    
    See :func:`save_tiff_folder` for saving a video as a folder of ``.tiff`` images.

    See :func:`export_video` for exporting a video to a movie file (e.g. ``.mp4``).

    Parameters
    ----------
    video : np.ndarray
        Video to save
    filename : str or pathlib.Path
        Filename to save to
    **kwargs : dict
        Additional arguments to pass to the respective save function
    """
    filename = Path(filename)
    suffix = filename.suffix.lower()
    if not suffix:
        raise RuntimeError(f"File '{filename}' has no file extension, unable to save video")

    if suffix in ['.tif', '.tiff']:
        save_tiff(video, filename)
    elif suffix in ['.npy']:
        np.save(filename, video, **kwargs)
        print(f'saved video as numpy file to {filename}')
    elif suffix in ['.mat']:
        save_matlab(video, filename, **kwargs)
    else:
        raise RuntimeError(f"Unrecognized file extension {suffix} (for file '{filename}')")
