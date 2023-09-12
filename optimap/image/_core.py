from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..utils import _print

def show_image(image, title="", vmin=None, vmax=None, cmap="gray", ax=None, **kwargs):
    """
    Show an image.

    Parameters
    ----------
    image : 2D ndarray
        Image to show.
    title : str, optional
        Title of the image, by default ""
    vmin : float, optional
        Minimum value for the colorbar, by default None
    vmax : float, optional
        Maximum value for the colorbar, by default None
    cmap : str, optional
        Colormap to use, by default "gray"
    ax : `matplotlib.axes.Axes`, optional
        Axes to plot on. If None, a new figure and axes is created.
    **kwargs : passed to :func:`matplotlib.pyplot.imshow`

    Returns
    -------
    ax : `matplotlib.axes.Axes`
    """
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False
    
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title)
    ax.axis("off")

    if show:
        plt.show()
    return ax

def load_image(filename, as_gray=False, **kwargs):
    """
    Load an image from a file. Eg. PNG, TIFF, NPY, ...
    
    Uses :func:`numpy.load` internally if the file extension is ``.npy``.
    Uses :func:`cv2.imread` internally otherwise.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename of image file to load (e.g. PNG, TIFF, ...)
    as_gray : bool, optional
        If True, convert color images to gray-scale. By default False.
    **kwargs : passed to :func:`cv2.imread` or :func:`numpy.load`

    Returns
    -------
    np.ndarray
        Image array, color images are in RGB(A) format
    """
    fn = Path(filename)
    _print(f'loading image from {fn.absolute()} ... ')

    if fn.suffix == '.npy':
        image = np.load(filename, **kwargs)
    else:
        # image = skimage.io.imread(filename, as_gray=as_gray, **kwargs)
        image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if as_gray and image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
    _print(f'Image shape: {image.shape[0]}x{image.shape[1]} pixels')
    return image

def load_mask(filename, **kwargs):
    """
    Load a mask from an image file.

    If the image is grayscale or RGB, the half of the maximum value is used as threshold. Values below the threshold are set to False, values above to True.

    If the image has 4 channels, the alpha channel is used as mask, with values below 0.5 set to False, and values above to True.

    Parameters
    ----------
    filename : str
        Filename of image file to load (e.g. NPY, PNG, TIFF, ...)
    **kwargs : passed to :func:`load_image`
    
    Returns
    -------
    np.ndarray of bool
        Mask array
    """
    mask = load_image(filename, **kwargs)
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = np.max(mask, axis=2)
    mask = mask < np.max(mask) / 2
    return mask

def save_image(image, filename, **kwargs):
    """
    Export an image to a file. The file format is inferred from the filename extension.

    Uses :func:`numpy.save` internally if the file extension is ``.npy``.
    Uses :func:`cv2.imwrite` internally otherwise.

    Parameters
    ----------
    image : {X, Y}, {X, Y, 3}, or {X, Y, 3} np.ndarray
        Image to save
    filename : str or pathlib.Path
        Path to save image to
    **kwargs : passed to :func:`cv2.imwrite`
    """
    _print(f"saving image to {Path(filename).absolute()}")
    fn = Path(filename)
    if fn.suffix == '.npy':
        np.save(filename, image, **kwargs)
    else:
        # skimage.io.imsave(filename, image, **kwargs)
        cv2.imwrite(filename, image, **kwargs)