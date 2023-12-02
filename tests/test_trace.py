import numpy as np
import pytest

import optimap as om


def test_extract_trace_rect():
    video = np.random.random((100, 100, 100))
    trace = om.extract_traces(video, [25, 50], size=1)
    assert trace.shape == (100,)
    assert np.allclose(trace, video[:, 50, 25])

    trace = om.extract_traces(video, [25, 50], size=3)
    x = video[:, 49:52, 24:27]
    assert x.shape == (100, 3, 3)
    assert np.allclose(trace, x.mean(axis=(1, 2)))

    trace = om.extract_traces(video, [25, 50], size=2)
    x = video[:, 49:51, 24:26]
    assert x.shape == (100, 2, 2)
    assert np.allclose(trace, x.mean(axis=(1, 2)))

def test_extract_trace_bounds():
    video = np.random.random((100, 100, 100))
    actual = om.extract_traces(video, [0, 0], size=5)
    expected = video[:, :3, :3].mean(axis=(1, 2))
    assert np.allclose(actual, expected)

    actual = om.extract_traces(video, [99, 99], size=5)
    expected = video[:, -3:, -3:].mean(axis=(1, 2))
    assert np.allclose(actual, expected)

def test_extract_trace_idx_outside():
    video = np.random.random((100, 100, 100))
    pytest.raises(ValueError, om.extract_traces, video, [-1, -1], size=0)
    pytest.raises(ValueError, om.extract_traces, video, [100, 100], size=0)

def test_extract_trace_pixel():
    video = np.random.random((100, 100, 100))
    trace = om.extract_traces(video, [25, 50], size=0)
    trace2 = om.extract_traces(video, [25, 50], size=5, window="pixel")
    trace3 = video[:, 50, 25]
    assert np.allclose(trace, trace2)
    assert np.allclose(trace, trace3)

def test_disc():
    video = np.zeros((100, 100, 100))
    video[:, 50:, :50] = 1
    trace = om.extract_traces(video, [20, 60], size=5, window="disc")
    assert np.allclose(trace, 1)
    trace2 = om.extract_traces(video, [20, 60], size=5, window="rect")
    assert np.allclose(trace, trace2)
