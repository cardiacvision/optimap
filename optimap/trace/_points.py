import numpy as np
import skimage


def random_positions(shape_or_mask, N=1):
    """Returns N randomly distributed positions within image shape or mask.

    Parameters
    ----------
    shape_or_mask : tuple or ndarray
        either image shape (e.g. (128, 128)) or a image mask
    N : int
        number of positions, by default 1

    Returns
    -------
    list of tuples
        list of N positions (y, x)
    """
    if isinstance(shape_or_mask, np.ndarray):
        mask = shape_or_mask.astype(bool)
        shape = mask.shape
    else:
        mask = np.full(shape_or_mask, True, dtype=bool)
        shape = shape_or_mask
    mask = mask / mask.sum()

    rng = np.random.default_rng()
    idxs = rng.choice(
        np.arange(mask.size),
        p=mask.flatten(),
        size=N,
        replace=False
    )
    points = np.unravel_index(idxs, shape)
    points = np.transpose(points)
    points = points[:, ::-1]
    return points.tolist()


def line_positions(start, end):
    return skimage.draw.line(start[0], start[1], end[0], end[1]) #TODO (y, x)?


def positions_from_A_to_B(N, A, B):
    """Creates N positions along straight line from A to B (2D coordinates).

    Parameters
    ----------
    N : int
        number of positions to create
    A : tuple
        starting point (x, y)
    B : tuple
        end point (x, y)

    Returns
    -------
    list of tuples
        TODO: (x, y) or (y, x) ???
    """
    msg = "nothing implemented yet"
    raise NotImplementedError(msg)
