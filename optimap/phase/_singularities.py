import numpy as np


def _normalize(p):
    """Normalize/wrap the phase differences `p` to values between -pi and pi."""
    m = abs(p) > np.pi
    s = np.sign(p[m])
    p[m] = -2 * s * np.pi + p[m]
    return p


def detect_phase_singularities(phase: np.ndarray, separate_by_charge=False):
    """Detect phase singularities in a phase map video using :cite:p:`Iyer2001` method.
    Unlike :cite:p:`Iyer2001`, the phase integral used here is only over four pixels.

    TODO: verify if positions are (x, y) or (y, x)

    Parameters
    ----------
    phase : ndarray
        Phase map video, shaped {t, x, y}
    separate_by_charge : bool, optional
        Whether to separate phase singularities by charge, by default False
        See return value for details.

    Returns
    -------
    list of tuples
        List of phase singularities for each frame (if separate_by_charge is False)
    (list of tuples, list of tuples)
        (positive_ps_list, negative_ps_list) (if separate_by_charge is True)
    """
    if phase.ndim != 3:
        msg = "phase must be three dimensional, shaped (t, x, y)"
        raise ValueError(msg)

    phase_singularities = []
    if separate_by_charge:
        phase_singularities = [[], []] # positive, negative

    # Loop over each frame:
    for indx in np.ndindex(*phase.shape[:-2]):
        phi = phase[indx]

        dx = _normalize(phi[1:, :] - phi[:-1, :])
        dy = _normalize(phi[:, 1:] - phi[:, :-1])

        # Use an integral around one node (i.e. through 4 pixels)
        summed = dx[:, 1:]
        summed += -dy[1:, :]
        summed += -dx[:, :-1]
        summed += dy[:-1, :]

        # Find phase singularities
        abs_summed = abs(summed)
        ps_positions = (abs_summed > np.pi) & (abs_summed < 3 * np.pi)
        signs = np.sign(summed[ps_positions])
        # PS positions are at the center of the 4 pixels
        ps_positions = np.array(np.where(ps_positions)).T + 0.5

        if separate_by_charge:
            phase_singularities[0].append(ps_positions[signs > 0])
            phase_singularities[1].append(ps_positions[signs < 0])
        else:
            ps = ps_positions[signs > 0].tolist()
            ps.extend(ps_positions[signs < 0].tolist())
            phase_singularities.append(ps)
    return phase_singularities
