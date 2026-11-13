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


def test_extract_trace_numpy_coords():
    video = np.random.random((10, 20, 30))
    expected = video[:, 4, 3]
    assert np.allclose(om.extract_traces(video, np.array([3, 4]), size=1), expected)
    assert np.allclose(om.extract_traces(video, (np.int64(3), np.int64(4)), size=1), expected)
    traces = om.extract_traces(video, np.array([[3, 4], [5, 6]]), size=1)
    assert traces.shape == (10, 2)


def test_collage_positions():
    positions = [np.array([[1, 2]]) for _ in range(7)]
    original = [p.copy() for p in positions]
    # images with 10 rows and 20 columns, positions in (column, row) order
    out = om.trace.collage_positions(positions, (10, 20), ncols=6)
    assert len(out) == 7
    assert np.array_equal(out[0], [1, 2])
    assert np.array_equal(out[1], [21, 2])  # second image in the first row
    assert np.array_equal(out[6], [1, 12])  # first image in the second row
    assert all(np.array_equal(p, o) for p, o in zip(positions, original))  # input not modified

    out = om.trace.collage_positions(positions[:2], (10, 20), ncols=1, padding=5)
    assert np.array_equal(out[1], [1, 17])

    # consistent with image collage
    images = [np.zeros((10, 20)) for _ in range(7)]
    for img, pos in zip(images, positions):
        img[pos[0, 1], pos[0, 0]] = 1
    collage = om.image.collage(images, ncols=6, padding=3)
    for x, y in om.trace.collage_positions(positions, (10, 20), ncols=6, padding=3):
        assert collage[y, x] == 1


def test_show_traces_array_colors(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    traces = np.random.random((10, 2))
    om.show_traces(traces, colors=np.array(["red", "blue"]), labels=np.array(["a", "b"]))
    om.show_traces(traces, colors="red")
    plt.close("all")

