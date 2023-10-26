import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage

from ._GHT import im2hist, GHT


def detect_background_threshold(img):
    """
    Detect the background threshold of an image using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    img : np.ndarray
        Image to detect background threshold for.

    Returns
    -------
    float or int
        Background threshold.
    """
    # TODO: fix histogram binning for float images properly
    scale = 1
    if img.dtype in [np.float32, np.float64]:
        max_val = img.max()
        if max_val < 5:
            scale = 4095
            img = img * 4095
    n, x = im2hist(img, zero_extents=True)
    threshold = GHT(n, x)[0]
    return threshold / scale


def background_mask(image, threshold=None, show=True, return_threshold=False):
    """
    Create a background mask for an image using a threshold.
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
        show_mask(mask, image)
    if return_threshold:
        return mask, threshold
    else:
        return mask


def foreground_mask(image, threshold=None, show=True, return_threshold=False):
    """
    Create a foreground mask for an image using thresholding.
    If no threshold is given, the background threshold is detected using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    image : 2D ndarray
        Image to create foreground mask for.
    threshold : float or int, optional
        Background threshold, by default None
    show : bool, optional
        Show the mask, by default True

    Returns
    -------
    mask : 2D ndarray
        Foreground mask.
    """

    if threshold is None:
        threshold = detect_background_threshold(image)
        print(f"Creating mask with detected threshold {threshold}")

    mask = image > threshold
    if show:
        show_mask(mask, image)
    if return_threshold:
        return mask, threshold
    else:
        return mask


def show_mask(mask, image=None, ax=None):
    """
    Show an mask overlayed on an image.
    If no image is given, only the mask is shown.

    Parameters
    ----------
    mask : 2D ndarray
        Mask to overlay.
    image : 2D ndarray
        Image to show.
    ax : `matplotlib.axes.Axes`, optional
        Axes to plot on. If None, a new figure and axes is created.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        Axes object with image and mask plotted."""
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    if image is not None:
        ax.imshow(image, cmap="gray", interpolation="none")
    alpha = 0.5 if image is not None else 1
    ax.imshow(mask, cmap="jet", alpha=alpha, interpolation="none")
    ax.axis("off")

    if show:
        plt.show()
    return ax


def disc_mask(shape, center, radius):
    """
    Create a circular/disk shaped mask.

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
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return mask


def largest_mask_island(mask: np.ndarray, invert: bool = False):
    """
    Identify and return the largest connected component (island) in a given binary mask.

    This function labels distinct regions (or islands) in a binary mask and returns the largest one in terms of pixel count. Optionally, the mask can be inverted before processing and then inverted back before returning the result.

    Parameters
    ----------
    mask : 2D ndarray
        Binary mask.
    invert : bool, optional
        If set to True, the mask is inverted before processing and then inverted back before returning the result. This can be useful if the area of interest is represented by False in the original mask. Default is False.

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
    return new_mask


def erode_mask(binary_mask, iterations=1, **kwargs):
    """
    Erode a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    **kwargs : dict, optional
        Additional arguments passed to `scipy.ndimage.binary_erosion`.

    Returns
    -------
    2D ndarray
        Eroded mask.
    """
    return ndimage.binary_erosion(binary_mask, iterations=iterations, **kwargs)


def dilate_mask(binary_mask, iterations=1, **kwargs):
    """
    Dilate a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    **kwargs : dict, optional
        Additional arguments passed to `scipy.ndimage.binary_dilation`.

    Returns
    -------
    2D ndarray
        Dilated mask.
    """
    return ndimage.binary_dilation(binary_mask, iterations=iterations, **kwargs)


def fill_mask_holes(binary_mask, **kwargs):
    """
    Fill holes in a binary mask.

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    **kwargs : dict, optional
        Additional arguments passed to `scipy.ndimage.binary_fill_holes`.

    Returns
    -------
    2D ndarray
        Mask with filled holes.
    """
    return ndimage.binary_fill_holes(binary_mask, **kwargs)


def binary_opening(binary_mask, iterations=1, **kwargs):
    """
    Perform binary opening on a binary mask. Consists of an erosion followed by a dilation. See https://en.wikipedia.org/wiki/Opening_(morphology).

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    **kwargs : dict, optional
        Additional arguments passed to `scipy.ndimage.binary_opening`.

    Returns
    -------
    2D ndarray
        Mask after binary opening.
    """
    return ndimage.binary_opening(binary_mask, iterations=iterations, **kwargs)


def binary_closing(binary_mask, iterations=1, **kwargs):
    """
    Perform binary closing on a binary mask. Consists of a dilation followed by an erosion. See https://en.wikipedia.org/wiki/Closing_(morphology).

    Parameters
    ----------
    binary_mask : 2D ndarray
        Binary mask.
    iterations : int, optional
        Number of iterations, by default 1
    **kwargs : dict, optional
        Additional arguments passed to `scipy.ndimage.binary_closing`.

    Returns
    -------
    2D ndarray
        Mask after binary closing.
    """
    return ndimage.binary_closing(binary_mask, iterations=iterations, **kwargs)