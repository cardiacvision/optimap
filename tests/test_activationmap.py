import pytest

import optimap as om
import numpy as np

def test_activationmap():
    vid = np.zeros((100, 100, 100), dtype=np.float32)
    for i in range(100):
        vid[i, i, i] = 1
    map = om.activation.compute_activation_map(vid, fps=100, show=False)
    assert map.shape == (100, 100)
    for i in range(100):
        assert map[i, i] == pytest.approx(i * 1000.0 / 100.0)
        map[i,i] = np.nan
    assert np.all(np.isnan(map))