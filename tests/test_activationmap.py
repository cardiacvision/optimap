import numpy as np
import pytest

import optimap as om


def test_activationmap():
    video = np.zeros((100, 100, 100), dtype=np.float32)
    video[10:, 0, 0] = 1
    for i in range(1, 100):
        video[60:, i, i] = 1
    amap = om.activation.compute_activation_map(video, show=False, method="threshold_crossing")
    assert amap[0, 0] == pytest.approx(0)
    for i in range(1, 100):
        assert amap[i, i] == pytest.approx(50)
    amap = om.activation.compute_activation_map(video, normalize_time=False, show=False, method="threshold_crossing")
    assert amap[0, 0] == 10
    for i in range(1, 100):
        assert amap[i, i] == pytest.approx(60)
    for i in range(0, 100):
        amap[i,i] = np.nan
    assert np.all(np.isnan(amap))

    amap = om.activation.compute_activation_map(video, show=False, method="maximum_derivative")
    assert not np.isnan(amap).any()

def test_find_activations():
    trace = np.zeros(100, dtype=np.float32)
    activations = om.activation.find_activations(trace, show=False)
    assert activations.size == 0

    trace[50] = 1
    activations = om.activation.find_activations(trace, method='threshold_crossing', min_duration=2, show=False)
    assert activations.size == 0

    trace[50:55] = 1
    trace[75:80] = 1
    activations = om.activation.find_activations(trace, method='threshold_crossing', min_duration=2, threshold=0.9, show=False)
    assert activations.size == 2
    assert activations[0] == 50
    assert activations[1] == 75

    trace = np.ones(100, dtype=np.float32)
    trace[50:55] = 0
    activations = om.activation.find_activations(trace, method='threshold_crossing', falling_edge=True, threshold=0.1, show=False)
    assert activations.size == 1
    assert activations[0] == 50
