import numpy as np

import optimap as om


def test_compute_phase():
    video = np.random.random((100, 10, 10)).astype(np.float32)
    phase = om.phase.compute_phase(video)
    assert phase.shape == video.shape
    assert phase.dtype == np.float32
    assert np.allclose(phase.min(), -np.pi, atol=0.2)
    assert np.allclose(phase.max(), np.pi, atol=0.2)
