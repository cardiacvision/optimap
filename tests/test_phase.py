import numpy as np

import optimap as om


def test_compute_phase():
    video = np.random.random((100, 10, 10)).astype(np.float32)
    phase = om.phase.compute_phase(video)
    assert phase.shape == video.shape
    assert phase.dtype == np.float32
    assert np.allclose(phase.min(), -np.pi, atol=0.2)
    assert np.allclose(phase.max(), np.pi, atol=0.2)


def test_phase_singularity_position_order():
    # single spiral centered between pixels at row 12.5, column 20.5 of a non-square image
    yy, xx = np.mgrid[:30, :40]
    phase = np.arctan2(yy - 12.5, xx - 20.5).astype(np.float32)[None]
    ps = om.phase.detect_phase_singularities(phase)
    assert ps == [[[12.5, 20.5]]]
    pos, neg = om.phase.detect_phase_singularities(phase, separate_by_charge=True)
    assert len(pos[0]) + len(neg[0]) == 1
