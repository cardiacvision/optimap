import numpy as np
import pytest

import optimap as om


def test_compute_cv_multiple_pairs():
    amap = np.tile(np.arange(50, dtype=np.float32), (40, 1))  # planar wave with 1 px/frame along x
    cv = om.compute_cv(amap, positions=[[(0, 0), (10, 0)], [(0, 5), (20, 5)]], show=False)
    assert np.allclose(cv, [1, 1])
    cv = om.compute_cv(amap, positions=[(0, 0), (30, 0)], fps=100, space_scale=0.1, show=False)
    assert cv == pytest.approx(100 * 0.1 / 10)  # px/frame -> cm/s


@pytest.mark.parametrize("method,kwargs", [
    ("bayly", {"window_size": 7}),
    ("circle", {"radius": 6}),
    ("circle", {"radius": 6, "angle_window_deg": 60}),
    ("gradient", {}),
])
def test_velocity_field_planar_wave(method, kwargs):
    speed = 2.0
    angle = np.deg2rad(30)
    yy, xx = np.mgrid[:60, :70].astype(np.float32)
    amap = (xx * np.cos(angle) + yy * np.sin(angle)) / speed
    vf = om.activation.compute_velocity_field(amap, method=method, **kwargs)
    assert np.nanmedian(np.linalg.norm(vf, axis=-1)) == pytest.approx(speed, rel=0.01)
    assert np.nanmedian(np.arctan2(vf[..., 1], vf[..., 0])) == pytest.approx(angle, abs=0.02)


def test_bayly_small_map_default_window():
    vf = om.activation.compute_velocity_field_bayly(np.random.rand(15, 15))
    assert vf.shape == (15, 15, 2)
