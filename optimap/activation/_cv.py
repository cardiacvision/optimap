import matplotlib.pyplot as plt
import numpy as np

from ._core import show_activation_map
from ..trace import select_positions
from ..utils import interactive_backend


@interactive_backend
def select_cv_positions(activation_map, **kwargs):
    cmap = kwargs.get("cmap", "turbo")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    positions = []
    while len(positions) != 2:
        positions = select_positions(activation_map, title="Select two points", num_points=2, cmap=cmap, vmin=vmin, vmax=vmax)
        if len(positions) != 2:
            print("Please select exactly two points.")
    return positions

def compute_cv(activation_map, positions=None, fps=None, space_scale=None, show=True, **kwargs):
    """Compute conduction velocity (CV) between pairs of positions on an activation map.

    The conduction velocity is calculated as the spatial distance between two points divided by the difference in activation times at those points.

    Parameters
    ----------
    activation_map : 2D array
        Activation map where values represent activation times in frames
    positions : list or ndarray, optional
        Position data in one of the following formats:
        
        - None: Interactive prompt to select two points (default)
        - List of N pairs of points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4), ...]
        - NumPy array with shape (N, 2, 2)
    fps : float, optional
        Frames per second. If provided, converts time units from frames to seconds
    space_scale : float, optional
        Spatial scale in mm/px. If provided, converts spatial units from pixels to cm
    show : bool, optional
        Whether to display a plot showing the activation map and computed CV, by default True
    **kwargs : dict, optional
        Additional keyword arguments passed to the `select_cv_positions` function if positions=None
        
    Returns
    -------
    float or ndarray
        Array of N conduction velocities.    
    
        Units depend on fps and space_scale:
          - pixels/frame if both fps and space_scale are None
          - pixels/s if only fps is provided
          - cm/s if both fps and space_scale are provided
    
    Examples
    --------
    >>> # Computing CV with user-selected points
    >>> cv = compute_cv(activation_map)
    >>>
    >>> # Computing CV with specific points in physical units
    >>> cv = compute_cv(activation_map, 
    ...                 positions=[(10, 15), (40, 30)],
    ...                 fps=500,  # 500 frames/s
    ...                 space_scale=10)  # 10 mm/px
    """
    if np.issubdtype(activation_map.dtype, np.integer):
        activation_map = activation_map.astype(np.float32)
    if positions is None:
        positions = select_cv_positions(activation_map, **kwargs)
    
    positions = np.array(positions)
    if positions.ndim == 2 and positions.shape[1] == 2:
        if positions.shape[0] == 2:
            positions = np.expand_dims(positions, axis=0)
        else:
            if positions.shape[0] % 2 != 0:
                raise ValueError("Number of positions should be multiple of 2.")
            positions = positions.reshape(-1, 2, 2)
    elif positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError("Positions should be a list of positions or an array of shape (N, 2, 2).")

    pos1 = positions[:, 0]
    pos2 = positions[:, 1]
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    time_diff = np.abs(activation_map[pos1[:, 1], pos1[:, 0]] - activation_map[pos2[:, 1], pos2[:, 0]])
    time_diff[time_diff == 0] = np.nan  # Avoid division by zero
    cv = distance / time_diff

    time_unit = "frame"
    space_unit = "pixels"
    if fps is not None:
        cv *= fps  # [px/frame] -> [px/s]
        time_unit = "s"
    if space_scale is not None:
        cv *= space_scale  # [px/s] -> [mm/s]
        cv /= 10  # [mm/s] -> [cm/s]
        space_unit = "cm"

    if show:
        fig, ax = plt.subplots()
        show_activation_map(activation_map, ax=ax)
        for i in range(len(positions)):
            x = [positions[i, 0, 0], positions[i, 1, 0]]
            y = [positions[i, 0, 1], positions[i, 1, 1]]
            ax.plot(x, y, "ro-")
        title = f"CV: {np.nanmean(cv):.2f} {space_unit}/{time_unit}"
        ax.set_title(title)
        plt.show()
    if len(cv) == 1:
        return cv.item()
    else:
        return cv
