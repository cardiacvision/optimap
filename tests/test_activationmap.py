import numpy as np
import pytest

import optimap as om


def test_activationmap():
    vid = np.zeros((100, 100, 100), dtype=np.float32)
    for i in range(100):
        vid[51, i, i] = 1
    amap = om.activation.compute_activation_map(vid, fps=100, show=False)
    assert amap.shape == (100, 100)
    for i in range(100):
        assert amap[i, i] == pytest.approx(50 * 1000.0 / 100.0)
        amap[i,i] = np.nan
    assert np.all(np.isnan(amap))

def test_find_activations():
    trace = np.zeros(100, dtype=np.float32)
    activations = om.activation.find_activations(trace, show=False)
    assert activations.size == 0

    trace[50] = 1
    activations = om.activation.find_activations(trace, min_duration=2, show=False)
    assert activations.size == 0

    trace[50:55] = 1
    trace[75:80] = 1
    activations = om.activation.find_activations(trace, min_duration=2, threshold=0.9, show=False)
    assert activations.size == 2
    assert activations[0] == 50
    assert activations[1] == 75

    trace = np.ones(100, dtype=np.float32)
    trace[50:55] = 0
    activations = om.activation.find_activations(trace, falling_edge=True, threshold=0.1, show=False)
    assert activations.size == 1
    assert activations[0] == 50
