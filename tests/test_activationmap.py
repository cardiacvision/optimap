import numpy as np
import pytest

import optimap as om


def test_activationmap():
    vid = np.zeros((100, 100, 100), dtype=np.float32)
    for i in range(100):
        vid[i, i, i] = 1
    amap = om.activation.compute_activation_map(vid, fps=100, show=False)
    assert amap.shape == (100, 100)
    for i in range(100):
        assert amap[i, i] == pytest.approx(i * 1000.0 / 100.0)
        amap[i,i] = np.nan
    assert np.all(np.isnan(amap))
