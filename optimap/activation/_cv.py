import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from ..image import show_image, smooth_gaussian
from ..trace import select_positions
from ..utils import interactive_backend
from ._core import show_activation_map


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
    time_diff[time_diff < 1e-6] = np.nan  # Avoid division by zero
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

def compute_velocity_field_bayly(activation_map, window_size=None, min_points_ratio=0.5):
    """
    Calculates a velocity vector field from a 2D activation map using local polynomial fitting (Bayly's method :cite:t:`Bayly1998`).

    For each point in the activation map, a local polynomial

    $$
    T(x,y) = ax^2 + by^2 + cxy + dx + ey + f
    $$

    is fitted to the activation times in a square window around that point. The velocity vector is then computed as the gradient of the fitted polynomial at the center of the window.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array where each element represents the activation time at that spatial location (grid point).
        NaN values can be used to indicate no activation.
    window_size : int
        The side length of the square window (neighborhood) used for local polynomial fitting.
        Must be an odd integer >= 3. Defaults to 10% of the image size.
    min_points_ratio : float
        The minimum fraction of non-NaN points required within a window to perform the fit (relative to window_size*window_size). Helps avoid fitting on sparse data. Defaults to 0.6.
    
    Returns
    -------
    np.ndarray
        Array of shape (rows, cols, 2) where the last dimension contains the x and y components of the velocity vector at each point.
    """
    if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
        raise ValueError("activation_map must be a 2D NumPy array.")
    if not isinstance(min_points_ratio, float) or not 0 < min_points_ratio <= 1:
         raise ValueError("min_points_ratio must be a float between 0 and 1.")

    rows, cols = activation_map.shape
    if window_size is None:
        window_size = min(rows, cols) // 10
        window_size += window_size % 2 == 0
    if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")

    half_window = window_size // 2
    min_points_count = int(np.ceil(window_size * window_size * min_points_ratio))
    num_coeffs = 6 # For T(x,y) = ax^2+by^2+cxy+dx+ey+f
    velocity_field = np.full(activation_map.shape + (2,), np.nan, dtype=np.float32)
    denom_threshold = 1e-6 # Avoid division by zero

    # Coordinate grid for the polynomial fitting
    rel_coords = np.arange(window_size) - half_window
    rel_xx, rel_yy = np.meshgrid(rel_coords, rel_coords)
    rel_xx_flat = rel_xx.flatten()
    rel_yy_flat = rel_yy.flatten()
    # Order of columns in A: x^2, y^2, xy, x, y, 1
    A_full = np.stack([
        rel_xx_flat**2, rel_yy_flat**2, rel_xx_flat * rel_yy_flat,
        rel_xx_flat, rel_yy_flat, np.ones_like(rel_xx_flat)
    ], axis=1)

    # Iterate and fit local polynomial
    for r in range(half_window, rows - half_window):
        for c in range(half_window, cols - half_window):
            # Skip if the center point is NaN
            if np.isnan(activation_map[r, c]):
                continue

            # Extract the activation times in the window
            t_window = activation_map[r - half_window : r + half_window + 1,
                                      c - half_window : c + half_window + 1]
            t_flat = t_window.flatten()
            valid_mask = ~np.isnan(t_flat)
            t_valid = t_flat[valid_mask]
            A_valid = A_full[valid_mask, :]

            if len(t_valid) < max(num_coeffs, min_points_count):
                continue

            try:
                # solve A * coeffs = t where coeffs = [a, b, c, d, e, f]
                coeffs, resid, rank, s = scipy.linalg.lstsq(A_valid, t_valid, cond=None)
            except np.linalg.LinAlgError:
                continue

            # Gradient at center (rel_x=0, rel_y=0): Tx = d, Ty = e
            Tx, Ty = coeffs[3], coeffs[4]
            denom = Tx**2 + Ty**2
            if denom < denom_threshold:
                continue

            # v = [Tx / (Tx^2 + Ty^2), Ty / (Tx^2 + Ty^2)]
            velocity_field[r, c, 0] = Tx / denom
            velocity_field[r, c, 1] = Ty / denom
    return velocity_field

def compute_velocity_field_circle(activation_map, radius=5, sigma=1, num_angles=180, angle_window_deg=30, min_valid_speeds_ratio=0.25):
    """
    Calculates a velocity vector field from a 2D activation map using the circle method :cite:t:`SilesParedes2022`.

    Velocity vectors represent direction and speed (pixels/frames) of propagation.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array (rows, cols) of local activation times (LAT). NaN values indicate masked areas.
    radius : float, optional.
        Radius of the circle used for LAT comparisons, in pixels. Defaults to 5px.
    sigma : float, optional
        Standard deviation for Gaussian smoothing applied to the absolute speeds on the circular kernel before finding the direction of propagation.
    num_angles : int, optional
        Number of diameters (angles from 0 to pi) to sample around the circle. Defaults to 180.
    angle_window_deg : float, optional
        Total angular window width (in degrees) centered around the estimated propagation direction,
        used for averaging speeds. Defaults to 30 degrees.
    min_valid_speeds_ratio : float, optional
        Minimum fraction of valid (non-NaN) instantaneous speeds required out of `num_angles`
        to compute a velocity vector at a point. Defaults to 0.25.

    Returns
    -------
    np.ndarray
        Array of shape (rows, cols, 2) where the last dimension contains the x and y components of the velocity vector at each point.
    """
    if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
        raise ValueError("activation_map must be a 2D NumPy array.")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("radius must be a positive number.")
    if not isinstance(angle_window_deg, (int, float)) or not 0 < angle_window_deg < 180:
        raise ValueError("angle_window_deg must be between 0 and 180 degrees.")
    if not isinstance(min_valid_speeds_ratio, float) or not 0 < min_valid_speeds_ratio <= 1:
         raise ValueError("min_valid_speeds_ratio must be a float between 0 and 1.")

    rows, cols = activation_map.shape
    velocity_field = np.full(activation_map.shape + (2,), np.nan, dtype=np.float32)
    min_valid_speeds_count = int(np.ceil(num_angles * min_valid_speeds_ratio))
    epsilon = 1e-9 # Avoid division by zero

    thetas = np.linspace(0, np.pi, num_angles, endpoint=False) # Angles for diameters (rad)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    angle_step_rad = np.pi / num_angles

    # Averaging window parameters
    half_window_rad = np.deg2rad(angle_window_deg) / 2.0
    w_indices = int(np.round(half_window_rad / angle_step_rad))
    relative_avg_angles_rad = np.arange(-w_indices, w_indices + 1) * angle_step_rad
    cos_relative_avg_angles = np.cos(relative_avg_angles_rad)
    # Prevent division by ~zero if window includes near pi/2 relative angle
    cos_relative_avg_angles[np.abs(cos_relative_avg_angles) < epsilon] = np.sign(cos_relative_avg_angles[np.abs(cos_relative_avg_angles) < epsilon]) * epsilon

    r_min, r_max = int(np.ceil(radius)), int(rows - np.ceil(radius))
    c_min, c_max = int(np.ceil(radius)), int(cols - np.ceil(radius))
    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            if np.isnan(activation_map[r, c]):
                continue

            # Get LATs on circumference
            r1 = r + radius * sin_thetas
            c1 = c + radius * cos_thetas
            r2 = r - radius * sin_thetas # Opposite point: angle + pi
            c2 = c - radius * cos_thetas
            coords_1 = np.vstack((r1, c1)) # Shape (2, num_angles)
            coords_2 = np.vstack((r2, c2))
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", UserWarning)
                 lat1 = scipy.ndimage.map_coordinates(activation_map, coords_1, order=1, mode="constant", cval=np.nan, prefilter=False)
                 lat2 = scipy.ndimage.map_coordinates(activation_map, coords_2, order=1, mode="constant", cval=np.nan, prefilter=False)

            # Instantaneous speeds
            delta_lat = lat1 - lat2
            speeds = 2.0 * radius / delta_lat
            abs_speeds = np.abs(speeds)
            if np.sum(~np.isnan(speeds)) < min_valid_speeds_count:
                continue  # Not enough valid points around the circle

            # Find propagation direction
            if sigma > 0:
                # Scipy's gaussian_filter does not support NaN values, do the same trick as for video smoothing
                smoothed_abs_speeds = abs_speeds.copy()
                smoothed_abs_speeds[np.isnan(abs_speeds)] = 0
                smoothed_abs_speeds = scipy.ndimage.gaussian_filter1d(smoothed_abs_speeds, sigma=sigma, mode="wrap")

                norm = np.ones_like(smoothed_abs_speeds, dtype=np.float32)
                norm[np.isnan(abs_speeds)] = 0
                norm = scipy.ndimage.gaussian_filter1d(norm, sigma=sigma, mode="wrap")
                smoothed_abs_speeds /= np.where(norm==0, 1, norm)
                smoothed_abs_speeds[np.isnan(abs_speeds)] = np.nan
            else:
                smoothed_abs_speeds = abs_speeds
            try:
                idx_min_abs_speed = np.nanargmin(smoothed_abs_speeds)
            except ValueError:
                continue

            # Compute average speed in the angle_window_deg window
            shift_amount = (-idx_min_abs_speed - num_angles // 2) % num_angles
            abs_speeds_oriented = np.roll(abs_speeds, shift_amount)
            window_abs_speeds = abs_speeds_oriented[num_angles // 2 - w_indices:num_angles // 2 + w_indices + 1]
            corrected_speeds = window_abs_speeds / cos_relative_avg_angles

            angle = thetas[idx_min_abs_speed] # radians
            if speeds[idx_min_abs_speed] < 0:
                angle += np.pi
            avg_speed_mag = np.nanmean(corrected_speeds)
            velocity_field[r, c, 0] = avg_speed_mag * np.cos(angle)
            velocity_field[r, c, 1] = avg_speed_mag * np.sin(angle)
    return velocity_field

def compute_velocity_field_gradient(activation_map, sigma=2, outlier_percentage=0):
    r"""Compute a velocity field based on the gradient of the activation map.

    This method estimates the velocity field by:
    1. Smoothing the input activation map using a Gaussian filter to reduce noise.
    2. Computing the velocity field :math:`\\nabla T / |\\nabla T|^2` of the smoothed
       activation time map :math:`T(x, y)`.
    3. Optionally remove outlier vectors based on gradient magnitude.

    Note: The result represents local velocity in pixels/frame.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array (rows, cols) where each pixel represents the local
        activation time (LAT). NaN values can be present.
    sigma : float, optional
        Standard deviation for the Gaussian smoothing applied to the
        activation map before gradient calculation. Larger values increase
        smoothing. Defaults to 2.0.
    outlier_percentage : float or None, optional
        The percentage of gradient vectors to filter out as outliers, based on
        their magnitude. Specifically, vectors whose gradient magnitude exceeds
        the `(100 - outlier_percentage)` percentile are set to NaN.
        For example, `outlier_percentage=0.1` removes the 0.1% of vectors with
        the largest magnitudes. Set to `None` or `0` to disable outlier removal.
        Defaults to 0.

    Returns
    -------
    np.ndarray
        3D NumPy array of shape (rows, cols, 2).

    See Also
    --------
    compute_velocity_field : General function calling different methods.
    """
    if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
        raise ValueError("activation_map must be a 2D NumPy array.")
    if not isinstance(sigma, (int, float)) or sigma < 0:
        raise ValueError("sigma must be a non-negative number.")
    
    activation_map = smooth_gaussian(activation_map, sigma=sigma)
    velocity_field = np.full(activation_map.shape + (2,), np.nan, dtype=np.float32)

    dy, dx = np.gradient(activation_map)
    gradient_magnitude_sq = dx**2 + dy**2
    gradient_magnitude_sq[gradient_magnitude_sq < 1e-6] = np.nan
    velocity_field[..., 0] = dx / gradient_magnitude_sq
    velocity_field[..., 1] = dy / gradient_magnitude_sq

    if outlier_percentage is not None and outlier_percentage > 0:
        threshold = np.nanpercentile(gradient_magnitude_sq, outlier_percentage)
        velocity_field[gradient_magnitude_sq < threshold] = np.nan
    return velocity_field

def compute_velocity_field(activation_map, method="bayly", **kwargs):
    r"""Computes the velocity field from an isochronal activation map.

    This function serves as a wrapper to call different algorithms for computing
    the velocity field, which represents the direction and magnitude (speed)
    of activation wavefront propagation at each point in the map.

    Available methods:
      * ``'bayly'``: Uses local second-order polynomial fitting to estimate the
        gradient of activation time and derive velocity. Use `window_size` parameter to control smoothing size. See :func:`compute_velocity_field_bayly` and :cite:t:`Bayly1998`.
      * ``'circle'``: Employs activation time differences across diameters of a
        local circle to determine velocity. Use `radius` parameter to control smoothing size. See :func:`compute_velocity_field_circle` and :cite:t:`SilesParedes2022`.
      * ``'gradient'``: Calculates velocity directly from the smoothed spatial
        gradient of the activation map (:math:`\vec{v} = \nabla T / |\nabla T|^2`).
        Simple and fast, but can be sensitive to noise and sharp gradients.
        See :func:`compute_velocity_field_gradient`.
    
    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array where each pixel represents the activation time.
    method : str
        Method to compute the velocity field. Options are: ['bayly', 'circle', 'gradient'].
    **kwargs : dict
        Additional parameters for the selected method.
    
    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array where each pixel represents the activation time.
    method : str
        Method to compute the velocity field. Options are: ['bayly', 'circle', 'gradient'].
    **kwargs : dict
        Additional parameters for the selected method.
    
    Returns
    -------
    np.ndarray
        3D NumPy array of shape (rows, cols, 2)
    """
    if method == "bayly":
        return compute_velocity_field_bayly(activation_map, **kwargs)
    elif method == "circle":
        return compute_velocity_field_circle(activation_map, **kwargs)
    elif method == "gradient":
        return compute_velocity_field_gradient(activation_map, **kwargs)
    else:
        raise ValueError(f"Method '{method}' is not supported. Available methods: 'bayly'.")

def compute_cv_map(activation_map, method="bayly", fps=None, space_scale=None, show=True, title=None, vmin=0, vmax=None, **kwargs):
    """
    Computes the local conduction velocity (CV) map from an activation map.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array where each pixel represents the activation time. NaN values mean no activation.
    method : str
        The method to use for computing the CV cmp. Currently only 'bayly' is supported.
    fps : float
        Frames per second. If provided, converts time units from frames to seconds.
    space_scale : float
        Spatial scale in mm/px. If provided, converts spatial units from pixels to cm.
    show : bool
        Whether to display a plot showing the activation map and computed CV map.

    Returns
    -------
    np.ndarray
        Array of shape (rows, cols, 2) where the last dimension contains the x and y components of the local CV at each point.

    See Also
    --------
    compute_cv: Computes the conduction velocity between pairs of points.
    compute_velocity_field : Computes the velocity field from an activation map.
    """
    velocity_field = compute_velocity_field(activation_map, method=method, **kwargs)
    cv = np.linalg.norm(velocity_field, axis=-1)

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
        title = f"Mean CV: {np.nanmean(cv):.2f} {space_unit}/{time_unit}" if title is None else title
        show_image(cv, title=title, cmap="turbo", vmin=vmin, vmax=vmax, show_colorbar=True, colorbar_title=f"CV [{space_unit}/{time_unit}]")
    return cv
