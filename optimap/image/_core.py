import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

from ..utils import _print
from ._edit import normalize


def show_image(image,
               title="",
               vmin=None,
               vmax=None,
               cmap="gray",
               show_colorbar=False,
               colorbar_title="",
               ax=None,
               **kwargs):
    """Show an image.

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
    show_colorbar : bool, optional
        Show colorbar on the side of the image, by default False
    colorbar_title : str, optional
        Label of the colorbar, by default ""
    ax : `matplotlib.axes.Axes`, optional
        Axes to plot on. If None, a new figure and axes is created.
    **kwargs : dict, optional
        passed to :func:`matplotlib.pyplot.imshow`

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        fig = ax.figure
        show = False

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title)
    ax.axis("off")

    if show_colorbar:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(colorbar_title)

    if show:
        plt.show()
    return ax

def load_image(filename, as_gray=False, **kwargs):
    """Load an image from a file. Eg. PNG, TIFF, NPY, ...

    Uses :func:`numpy.load` internally if the file extension is ``.npy``.
    Uses :func:`cv2.imread` internally otherwise.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename of image file to load (e.g. PNG, TIFF, ...)
    as_gray : bool, optional
        If True, convert color images to gray-scale. By default False.
    **kwargs : dict, optional
        passed to :func:`cv2.imread` or :func:`numpy.load`

    Returns
    -------
    np.ndarray
        Image array, color images are in RGB(A) format
    """
    fn = Path(filename)
    _print(f"loading image from {fn.absolute()} ... ")

    if fn.suffix == ".npy":
        image = np.load(fn, **kwargs)
    else:
        # image = skimage.io.imread(filename, as_gray=as_gray, **kwargs)
        image = cv2.imread(str(fn),
                           cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if as_gray and image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    _print(f"Image shape: {image.shape[0]}x{image.shape[1]} pixels")
    return image

def load_mask(filename, **kwargs):
    """Load a mask from an image file.

    Supports NPY, PNG, TIFF, ... files.

    If the image is grayscale or RGB, half of the maximum value is used as a threshold. Values below
    the threshold are set to ``False``, values above to ``True``.

    If the image is RGBA, then the alpha channel is used as mask using the same threshold.

    See :func:`save_mask` to save a mask to a file.

    Parameters
    ----------
    filename : str
        Filename of image file to load (e.g. NPY, PNG, TIFF, ...)
    **kwargs : dict, optional
        passed to :func:`load_image`

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
    if mask.dtype == bool:
        return mask
    else:
        return mask > np.max(mask) / 2


def save_mask(filename, mask, image=None, **kwargs):
    """Save a mask to a file.

    Supports NPY, PNG, TIFF, ... files.

    For NPY files, the mask is saved as a boolean array. For image files (PNG, TIFF, ...),
    the mask is saved as the alpha channel of the image (if given). If no image is given,
    the mask is saved as a grayscale image.

    See :func:`load_mask` to load the mask again.


    Parameters
    ----------
    filename : str or pathlib.Path
        Path to save mask to
    mask : np.ndarray
        2D bool mask to save
    image : np.ndarray, optional
        Image to save. If given, the mask will be saved as the alpha channel of the image.
        Only supported for .png and .tif files.
    **kwargs : dict, optional
        passed to :func:`np.save` (for .npy) or :func:`cv2.imwrite` (else)
    """
    if isinstance(mask, (str, os.PathLike)):
        filename, mask = mask, filename
        warnings.warn("The order of arguments for optimap.image.save_mask() has changed. "
                      "Please use save_mask(filename, mask, ...) instead of save_mask(mask, filename, ...).",
                      DeprecationWarning)

    mask = np.squeeze(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("mask must be 2D boolean array")

    if image is not None:
        image = np.squeeze(image)
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("image must be 2D or RBG(A)")

    suffix = Path(filename).suffix
    if suffix == ".npy":
        if image is not None:
            warnings.warn("save_mask() does not support saving images to .npy files", UserWarning)
        np.save(filename, mask, **kwargs)
    else:
        if image is not None:
            image = cv2.convertScaleAbs(image, alpha=(255.0/image.max()))  # convert to 8-bit
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            image[np.logical_not(mask), 3] = 0
            image[mask, 3] = 255
            cv2.imwrite(str(filename), image, **kwargs)
        else:
            mask = mask.astype(np.uint8) * 255
            cv2.imwrite(str(filename), mask, **kwargs)


def save_image(filename, image: np.ndarray, compat=False, **kwargs):
    """Save an image to a file. Makes best effort to avoid data precision loss, use :func:`export_image` to export images for publications.
    
    The image data is saved as it is, without any normalization or scaling.

    The following file formats and image data types are supported:
    * NumPy: .npy, all data types
    * PNG: .png, 8-bit or 16-bit unsigned per image channel
    * TIFF: .tif/.tiff, 8-bit unsigned, 16-bit unsigned, 32-bit float, or 64-bit float images
    * JPEG: .jpeg/.jpg, 8-bit unsigned
    * Windows bitmaps: .bmp, 8-bit unsigned

    The file format is inferred from the filename extension.

    Uses :func:`numpy.save` internally if the file extension is ``.npy`` and :func:`cv2.imwrite` otherwise.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to save image to
    image : np.ndarray
        Image to save
    compat : bool, optional
        If True, convert the image to a standard 8-bit format before saving. This can prevent issues where high-bitrate images appear black in some viewers. Defaults to False.
    **kwargs : dict, optional
        passed to :func:`cv2.imwrite`
    """
    if isinstance(image, (str, os.PathLike)):
        filename, image = image, filename
        warnings.warn("WARNING: The order of arguments to save_image has changed. "
                      "Please use save_image(filename, image) instead of save_video(image, filename).",
                      DeprecationWarning)

    _print(f"saving image to {Path(filename).absolute()}")
    fn = Path(filename)
    if fn.suffix == ".npy":
        np.save(fn, image, **kwargs)
    else:
        if compat:
            image = normalize(image, dtype="uint8")
        cv2.imwrite(str(fn), image, **kwargs)


def export_image(filename,
                 image: np.ndarray,
                 cmap = "gray",
                 vmin : float = None,
                 vmax : float = None):
    """Export an image to a file for publications, use :func:`save_image` to save an image if it will be reimported later.

    Images will be converted to uint8, colormap will be applied to grayscale images.

    The file format is inferred from the filename extension.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to save image to
    image : np.ndarray
        Image to save HxW or HxWxC
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for grayscale images, by default "gray"
    vmin : float, optional
        Minimum value for the colormap, by default None
    vmax : float, optional
        Maximum value for the colormap, by default None
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    if image.ndim == 2:
        image = cmap(norm(image))

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    save_image(filename, image)


def smooth_gaussian(image, sigma, **kwargs):
    """Smooth an image or mask using a Gaussian filter.

    Uses :func:`scipy.ndimage.gaussian_filter` internally with ``mode='nearest'``.

    Parameters
    ----------
    image : {X, Y} np.ndarray
        Image or mask to smooth
    sigma : float
        Standard deviation of the Gaussian kernel
    **kwargs : dict, optional
        passed to :func:`scipy.ndimage.gaussian_filter`
    """
    if "mode" not in kwargs:
        kwargs["mode"] = "nearest"

    if image.ndim == 3:  # RGB(A) image
        sigma = (sigma, sigma, 0)

    return ndimage.gaussian_filter(image, sigma=sigma, **kwargs)


def collage(images, ncols=6, padding=0, padding_value=0):
    """Create a collage from a list or array of images/masks that have the same shape.

    Creates a numpy array with the images arranged in a grid, with a specified number of images per row.
    Optionally, padding can be added between the images.

    See :func:`export_videos` to save a video collage to a video file.

    Parameters
    ----------
    images : np.ndarray or list of np.ndarray
        List of grayscale or RGB(A) images to combine
    ncols : int, optional
        Number of images per row, by default 6
    padding : int, optional
        Spacing between images in pixels, by default 0
    padding_value : float or np.ndarray, optional
        Value for the spacing (e.g. color RGB(A) array), by default 0

    Returns
    -------
    np.ndarray
        Collage image

    Examples
    --------
    .. code-block:: python

            import optimap as om
            import numpy as np

            # create a collage of 3x3 random images
            images = [np.random.rand(100, 100) for _ in range(9)]
            collage = om.image.collage(images, ncols=3, padding=10)
            om.image.show_image(collage)
    """
    for image in images:
        if image.shape != images[0].shape:
            raise ValueError("All images must have the same shape")

    pad_array = np.full((images[0].shape[0], padding) + images[0].shape[2:],
                         padding_value, dtype=images[0].dtype)

    collage_rows = []
    current_index = 0
    while current_index < len(images):
        end_index = min(current_index + ncols, len(images))
        row_images = [images[i] for i in range(current_index, end_index)]

        if end_index - current_index < ncols:
            blank_image = np.full(images[0].shape, padding_value, dtype=images[0].dtype)
            row_images.extend([blank_image] * (ncols - len(row_images)))

        if padding > 0:
            for i in range(len(row_images) - 1):
                row_images.insert(2 * i + 1, pad_array)

        collage_rows.append(np.hstack(row_images))
        current_index = end_index

    if padding > 0 and len(collage_rows) > 1:
        row_space = np.full((padding,) + collage_rows[0].shape[1:],
                            padding_value, dtype=collage_rows[0].dtype)
        for i in range(len(collage_rows) - 1):
            collage_rows.insert(2 * i + 1, row_space)

    collage_image = np.vstack(collage_rows)
    return collage_image
