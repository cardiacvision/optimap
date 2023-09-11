import numpy as np
import skimage

from ..utils import _print, print_bar


def random_positions(shape_or_mask, N=1):
    """
    Returns N randomly distributed positions within image shape or mask

    Parameters
    ----------
    shape_or_mask : tuple or ndarray
        either image shape (e.g. (128, 128)) or a image mask
    N : int
        number of positions, by default 1
    
    Returns
    -------
    list of tuples
        TODO: (x, y) or (y, x) ???
    """

    mask = None
    if isinstance(shape_or_mask, np.ndarray):
        mask = shape_or_mask
        Nx, Ny = mask.shape
        _print(f'creating {N} random (2D) positions within mask')
    else:
        Nx, Ny = shape_or_mask
        _print(f'creating {N} random (2D) positions within (0,0) and ({Nx}, {Ny})')

    points = []
    while len(points) < N:
        x = np.random.randint(Nx)
        y = np.random.randint(Ny)
        if (x, y) in points:
            continue

        # TODO: this is not very efficient, but works for now
        if mask is not None:
            if mask[x, y]: #TODO (y, x)?
                points.append((x, y))
        else:
            points.append((x, y))
    points = np.asarray(points)
    points = np.transpose(points)
    print_bar()
    return points

def line_positions(start, end):
    return skimage.draw.line(start[0], start[1], end[0], end[1]) #TODO (y, x)?

def positions_from_A_to_B(N, A, B):
    """
    Creates N positions along straight line from A to B (2D coordinates)
    
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
    raise NotImplementedError('nothing implemented yet')
