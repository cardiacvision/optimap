import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from ._core import show_activation_map
from ..trace import select_positions
from ..utils import interactive_backend
from ..image import show_image


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

def compute_velocity_field_bayly(activation_map, window_size=None, min_points_ratio=0.5):
    """
    Calculates a velocity vector field from a 2D activation map using local polynomial fitting (Bayly's method :cite:t:`Bayly1998`).

    For each point in the activation map, a local polynomial

    $$
    T(x,y) = ax^2 + by^2 + cxy + dx + ey + f
    $$

    is fitted to the activation times in a square window around that point. The velocity vector is then computed as the gradient of the fitted polynomial at the center of the window.

    % The conduction velocity vectors are highly dependent on the goodness of
    % fit of the polynomial surface.  In the Balyly paper, a 2nd order polynomial 
    % surface is used.  We found this polynomial to be insufficient and thus increased
    % the order to 3.  MATLAB's intrinsic fitting functions might do a better
    % job fitting the data and should be more closely examined if velocity
    % vectors look incorrect.

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
        window_size += window_size % 2
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

def compute_velocity_field_circle(activation_map, radius, num_angles=180, angle_window_deg=30, smooth_sigma_deg=10.0, min_valid_speeds_ratio=0.25):
    """
    Calculates a velocity vector field from a 2D activation map using the circle method :cite:t:`SilesParedes2022`.

    Velocity vectors represent direction and speed (pixels/frames) of propagation.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array (rows, cols) of local activation times (LAT). NaN values indicate masked areas.
    radius : float
        Radius of the circle used for LAT comparisons, in pixels.
    num_angles : int, optional
        Number of diameters (angles from 0 to pi) to sample around the circle. Defaults to 180.
    angle_window_deg : float, optional
        Total angular window width (in degrees) centered around the estimated propagation direction,
        used for averaging speeds. Defaults to 30 degrees.
    smooth_sigma_deg : float or None, optional
        Standard deviation (in degrees) for Gaussian smoothing applied to the absolute speeds before
        finding the minimum, used to robustly determine the propagation direction.
        Set to None or 0 to disable smoothing. Defaults to 10.0.
    min_valid_speeds_ratio : float, optional
        Minimum fraction of valid (non-NaN) instantaneous speeds required out of `num_angles`
        to compute a velocity vector at a point. Defaults to 0.25.

    Returns
    -------
    np.ndarray
        Array of shape (rows, cols, 2) where the last dimension contains the x and y components of the velocity vector at each point.
    """
    # --- Input Validation ---
    if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
        raise ValueError("activation_map must be a 2D NumPy array.")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("radius must be a positive number.")
    if not isinstance(num_angles, int) or num_angles <= 2:
        raise ValueError("num_angles must be an integer greater than 2.")
    if not isinstance(angle_window_deg, (int, float)) or not 0 < angle_window_deg < 180:
        raise ValueError("angle_window_deg must be between 0 and 180 degrees.")
    if smooth_sigma_deg is not None and (not isinstance(smooth_sigma_deg, (int, float)) or smooth_sigma_deg < 0):
        raise ValueError("smooth_sigma_deg must be a non-negative number or None.")
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

    # Smoothing parameter (convert degrees to points)
    if smooth_sigma_deg is not None and smooth_sigma_deg > epsilon:
        sigma_pts = smooth_sigma_deg / np.rad2deg(angle_step_rad)
    else:
        sigma_pts = 0

    # --- Iterate through pixels ---
    # Define calculation bounds considering the radius
    r_min, r_max = int(np.ceil(radius)), int(rows - np.ceil(radius))
    c_min, c_max = int(np.ceil(radius)), int(cols - np.ceil(radius))

    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            # Skip if the center point itself is invalid
            if np.isnan(activation_map[r, c]):
                continue

            # --- Get LATs on circumference using interpolation ---
            r1 = r + radius * sin_thetas
            c1 = c + radius * cos_thetas
            r2 = r - radius * sin_thetas # Opposite point: angle + pi
            c2 = c - radius * cos_thetas
            coords_1 = np.vstack((r1, c1)) # Shape (2, num_angles)
            coords_2 = np.vstack((r2, c2))

            # Use bilinear interpolation (order=1), handle boundaries with NaN
            with warnings.catch_warnings(): # Suppress map_coordinates boundary warnings
                 warnings.simplefilter("ignore", UserWarning)
                 lat1 = scipy.ndimage.map_coordinates(activation_map, coords_1, order=1, mode='constant', cval=np.nan, prefilter=False)
                 lat2 = scipy.ndimage.map_coordinates(activation_map, coords_2, order=1, mode='constant', cval=np.nan, prefilter=False)

            # --- Calculate instantaneous speeds (pixels/time_unit) ---
            delta_lat = lat1 - lat2
            valid_mask = ~np.isnan(delta_lat) & (np.abs(delta_lat) > epsilon)

            if np.sum(valid_mask) < min_valid_speeds_count:
                continue # Not enough valid points around the circle

            distance_pixels = 2.0 * radius # Diameter in pixels
            speeds = np.full(num_angles, np.nan, dtype=np.float32)
            speeds[valid_mask] = distance_pixels / delta_lat[valid_mask]
            abs_speeds = np.abs(speeds)

            # --- Find Propagation Direction ---
            # Smooth the *absolute* speeds to find the minimum robustly
            if sigma_pts > 0:
                smoothed_abs_speeds = scipy.ndimage.gaussian_filter1d(abs_speeds, sigma=sigma_pts, mode='wrap')
            else:
                smoothed_abs_speeds = abs_speeds

            # Find index of minimum absolute speed (handle potential all-NaNs)
            try:
                idx_min_abs_speed = np.nanargmin(smoothed_abs_speeds)
            except ValueError:
                continue

            # Propagation angle is orthogonal to the direction of min absolute speed
            angle_normal = thetas[idx_min_abs_speed] # radians
            original_speed_at_min = speeds[idx_min_abs_speed] # Use unsmoothed speed sign
            if np.isnan(original_speed_at_min): continue # Safety check

            if original_speed_at_min >= 0: # Wave moves from point 2 to point 1 along normal
                angle_prop = angle_normal + np.pi / 2.0
            else: # Wave moves from point 1 to point 2 along normal
                angle_prop = angle_normal - np.pi / 2.0

            angle_prop = angle_prop % (2 * np.pi) # Normalize angle to [0, 2*pi)

            # --- Calculate Average Speed Magnitude with Cosine Correction ---
            # Shift the *unsmoothed* absolute speeds so estimated prop direction is centered
            # Equivalent to MATLAB: circshift(abs(CV_f), -loc - L/4) where L/4 shifts by 90 deg
            shift_amount = (-idx_min_abs_speed - num_angles // 2) % num_angles
            abs_speeds_oriented = np.roll(abs_speeds, shift_amount)

            # Define the window slice around the center (index num_angles // 2)
            center_idx = num_angles // 2
            # Handle wrapping for the slice indices using modulo arithmetic
            indices_to_avg = np.mod(np.arange(center_idx - w_indices, center_idx + w_indices + 1), num_angles)

            # Extract the absolute speeds within the window & apply cosine correction
            window_abs_speeds = abs_speeds_oriented[indices_to_avg]
            corrected_speeds = window_abs_speeds / cos_relative_avg_angles

            avg_speed_mag = np.nanmean(corrected_speeds)
            if np.isnan(avg_speed_mag) or avg_speed_mag < 0:
                continue

            vx = avg_speed_mag * np.cos(angle_prop)
            vy = avg_speed_mag * np.sin(angle_prop)

            velocity_field[r, c, 0] = vx
            velocity_field[r, c, 1] = vy

    return velocity_field

def compute_velocity_field(activation_map, method='bayly', **kwargs):
    if method == 'bayly':
        return compute_velocity_field_bayly(activation_map, **kwargs)
    elif method == 'circle':
        return compute_velocity_field_circle(activation_map, **kwargs)
    else:
        raise ValueError(f"Method '{method}' is not supported. Available methods: 'bayly'.")

def compute_local_cv(activation_map, method='bayly', fps=None, space_scale=None, show=True, vmin=0, vmax=None, **kwargs):
    """
    Computes the local conduction velocity (CV) from an activation map using a specified method.

    Parameters
    ----------
    activation_map : np.ndarray
        2D NumPy array where each element represents the activation time at that spatial location (grid point).
        NaN values can be used to indicate no activation.
    method : str
        The method to use for computing the local CV. Currently only 'bayly' is supported.
    fps : float
        Frames per second. If provided, converts time units from frames to seconds.
    space_scale : float
        Spatial scale in mm/px. If provided, converts spatial units from pixels to cm.
    show : bool
        Whether to display a plot showing the activation map and computed local CV.

    Returns
    -------
    np.ndarray
        Array of shape (rows, cols, 2) where the last dimension contains the x and y components of the local CV at each point.
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
        title = f"Mean CV: {np.nanmean(cv):.2f} {space_unit}/{time_unit}"
        show_image(cv, title=title, cmap="turbo", vmin=vmin, vmax=vmax)
    return cv
