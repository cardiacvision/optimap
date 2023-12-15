import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage

from ..utils import interactive_backend
from ._GHT import GHT, im2hist
from ._segmenter import ImageSegmenter


@interactive_backend
def interactive_mask(image, initial_mask=None, default_tool="draw", cmap="gray", title="", figsize=(7, 7)):
    """Create a mask interactively by drawing on an image.

    .. table:: **Keyboard Shortcuts**

        ========================= ===========================
        Key                       Action
        ========================= ===========================
        ``Scroll``                Zoom in/out
        ``ctrl+z`` or ``cmd+z``   Undo
        ``ctrl+y`` or ``cmd+y``   Redo
        ``e``                     Erase mode
        ``d``                     Draw/Lasso mode
        ``v``                     Toggle mask visibility
        ``q``                     Quit
        ========================= ===========================


    Parameters
    ----------
    image : 2D ndarray
        Image to draw on.
    initial_mask : 2D ndarray, optional
        Mask to start with, by default None
    default_tool : str, optional
        Default tool to use, by default "draw". Can be one of ``draw`` or ``erase``.
    cmap : str, optional
        Colormap of the image, by default "gray"
    title : str, optional
        Title of the image, by default ""
    figsize: tuple, optional
        Figure size, by default (7, 7)

    Returns
    -------
    mask : 2D ndarray
        Created mask.
    """
    fig, ax = plt.subplots(figsize=figsize)
    segmenter = ImageSegmenter(
        image,
        mask=initial_mask,
        default_tool=default_tool,
        cmap=cmap,
        ax=ax,
        title=title,
    )
    plt.show(block=True)
    return segmenter.mask


def detect_background_threshold(image):
    """Detect the background threshold of an image using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    image : np.ndarray
        Image to detect background threshold for.

    Returns
    -------
    float or int
        Background threshold.
    """
    # TODO: fix histogram binning for float images properly

    scale = 1
    if image.dtype in [np.float32, np.float64]:
        max_val = image.max()
        if max_val < 5:
            scale = 4095
            image = image * 4095
    n, x = im2hist(image, zero_extents=True)
    threshold = GHT(n, x)[0]
    return threshold / scale


def background_mask(image, threshold=None, show=True, return_threshold=False, **kwargs):
    """Create a background mask for an image using a threshold.

    If no threshold is given, the background threshold is detected using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    image : 2D ndarray
        Image to create background mask for.
    threshold : float or int, optional
        Background threshold, by default None
    show : bool, optional
        Show the mask, by default True
    return_threshold : bool, optional
        If True, return the threshold as well, by default False
    kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    mask : 2D ndarray
        Background mask.
    threshold : float or int
        Background threshold, only if ``return_threshold`` is True.
    """
    if threshold is None:
        threshold = detect_background_threshold(image)
        print(f"Creating mask with detected threshold {threshold}")

    mask = image < threshold
    if show:
        show_mask(mask, image, **kwargs)
    if return_threshold:
        return mask, threshold
    else:
        return mask


def foreground_mask(image, threshold=None, show=True, return_threshold=False, **kwargs):
    """Create a foreground mask for an image using thresholding.

    If no threshold is given, the background threshold is detected using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    image : 2D ndarray
        Image to create foreground mask for.
    threshold : float or int, optional
        Background threshold, by default None
    show : bool, optional
        Show the mask, by default True
    return_threshold : bool, optional
        If True, return the threshold as well, by default False
    kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    mask : 2D ndarray
        Foreground mask.
    threshold : float or int
        Background threshold, only if ``return_threshold`` is True.
    """
    if threshold is None:
        threshold = detect_background_threshold(image)
        print(f"Creating mask with detected threshold {threshold}")

    mask = image > threshold
    if show:
        show_mask(mask, image, **kwargs)
    if return_threshold:
        return mask, threshold
    else:
        return mask


def show_mask(mask, image=None, title="", alpha=0.3, color="red", cmap="gray", ax=None):
    """Show an mask overlayed on an image.
    If no image is given, only the mask is shown.

    Parameters
    ----------
    mask : 2D ndarray
        Mask to overlay.
    image : 2D ndarray
        Image to show.
    title : str, optional
        Title of the image, by default ""
    alpha : float, optional
        Alpha value of the mask, by default 0.5
    color : str, optional
        Color of the True values of the mask, by default 'red'
    cmap : str, optional
        Colormap of the image, by default "gray"
    ax : `matplotlib.axes.Axes`, optional
        Axes to plot on. If None, a new figure and axes is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with image and mask plotted.
    """
    if mask.ndim != 2:
        msg = f"Mask must be an image, got shape {mask.shape}"
        raise ValueError(msg)
    if image is not None and image.ndim != 2 and image.shape[-1] != 3:
        msg = f"Image must be an image, got shape {image.shape}"
        raise ValueError(msg)
    mask = mask.astype(bool)
    cmap_mask = mpl.colors.ListedColormap(["none", color])

    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    if image is not None:
        ax.imshow(image, cmap=cmap, interpolation="none")
    else:
        alpha = 1

    ax.imshow(mask, cmap=cmap_mask, vmin=0, vmax=1, alpha=alpha, interpolation="none")

    if image is not None:
        ax.axis("off")
    else:  # show border around image
        ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)

    if title:
        ax.set_title(title)

    if show:
        plt.show()
    return ax


def invert_mask(mask):
    """Invert a binary mask.

    Parameters
    ----------
    mask : 2D ndarray
        Binary mask.

    Returns
    -------
    2D ndarray
        Inverted mask.
    """
    return np.logical_not(mask)

def disc_mask(shape, center, radius):
    """Create a circular/disk shaped mask.

    Parameters
    ----------
    shape : tuple
        Shape of the mask.
    center : tuple
        Center of the circle in (x, y) format. TODO: check if this is correct, or if it should be (y, x)?
    radius : float
        Radius of the circle.

    Returns
    -------
    mask : 2D ndarray
        Mask.
    """
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

def largest_mask_component(mask: np.ndarray, invert: bool = False, show=True, **kwargs):
    """Identify and return the largest connected component (island) in a given binary mask.

    This function labels distinct unconnected regions (or islands) in a binary mask and returns the largest
    one in terms of pixel count. Optionally, the mask can be inverted before processing and then inverted
    back before returning the result.

    Parameters
    ----------
    mask : 2D ndarray
        Binary mask.
    invert : bool, optional
        If set to True, the mask is inverted before processing and then inverted back before returning the result.
        This can be useful if the area of interest is represented by False in the original mask. Default is False.
    show : bool, optional
        Show the resulting mask, by default True
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Largest island.
    """
    if invert:
        mask = np.logical_not(mask)

    labels = skimage.measure.label(mask)
    label_count = np.bincount(labels.ravel())
    largest_label = label_count[1:].argmax() + 1
    new_mask = labels == largest_label

    if invert:
        new_mask = np.logical_not(new_mask)

    if show:
        show_mask(new_mask, **kwargs)
    return new_mask


def erode_mask(binary_mask, iterations=1, border_value=0, structure=None, show=True, **kwargs):
    """Erode a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    border_value : int, optional
        Value of the border, by default 0
    show : bool, optional
        Show the resulting mask, by default True
    structure : array_like, optional
        Structuring element used for the erosion. See
        `scipy documentation <https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#morphology>`_
        for more information.
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Eroded mask.
    """
    mask = ndimage.binary_erosion(binary_mask, iterations=iterations, structure=structure, border_value=border_value)
    if show:
        show_mask(mask, **kwargs)
    return mask


def dilate_mask(binary_mask, iterations=1, border_value=0, structure=None, show=True, **kwargs):
    """Dilate a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    border_value : int, optional
        Value of the border, by default 0
    structure : array_like, optional
        Structuring element used for the erosion. See
        `scipy documentation <https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#morphology>`_
        for more information.
    show : bool, optional
        Show the resulting mask, by default True
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Dilated mask.
    """
    mask = ndimage.binary_dilation(binary_mask, iterations=iterations, border_value=border_value, structure=structure)
    if show:
        show_mask(mask, **kwargs)
    return mask


def fill_mask(binary_mask, structure=None, show=True, **kwargs):
    """Fill holes in a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    structure : array_like, optional
        Structuring element used for the erosion. See
        `scipy documentation <https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#morphology>`_
        for more information.
    show : bool, optional
        Show the resulting mask, by default True
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Mask with filled holes.
    """
    mask = ndimage.binary_fill_holes(binary_mask, structure=structure)
    if show:
        show_mask(mask, **kwargs)
    return mask


def open_mask(binary_mask, iterations=1, border_value=0, structure=None, show=True, **kwargs):
    """Perform binary opening on a binary mask. Consists of an erosion followed by a dilation.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    border_value : int, optional
        Value of the border, by default 0
    structure : array_like, optional
        Structuring element used for the erosion. See
        `scipy documentation <https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#morphology>`_
        for more information.
    show : bool, optional
        Show the resulting mask, by default True
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Mask after binary opening.
    """
    mask = ndimage.binary_opening(binary_mask,
                                  iterations=iterations,
                                  border_value=border_value,
                                  structure=structure)
    if show:
        show_mask(mask, **kwargs)
    return mask


def close_mask(binary_mask, iterations=1, border_value=0, structure=None, show=True, **kwargs):
    """Perform binary closing on a binary mask. Consists of a dilation followed by an erosion.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    border_value : int, optional
        Value of the border, by default 0
    structure : array_like, optional
        Structuring element used for the erosion. See
        `scipy documentation <https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#morphology>`_
        for more information.
    show : bool, optional
        Show the resulting mask, by default True
    **kwargs : dict, optional
        Additional arguments passed to :func:`show_mask`.

    Returns
    -------
    2D ndarray
        Mask after binary closing.
    """
    mask = ndimage.binary_closing(binary_mask,
                                  iterations=iterations,
                                  border_value=border_value,
                                  structure=structure)
    if show:
        show_mask(mask, **kwargs)
    return mask
