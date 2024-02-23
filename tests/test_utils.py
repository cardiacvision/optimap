from pathlib import Path

import numpy as np

import optimap as om


def test_download_example_data():
    filename = om.utils.download_example_data("test-download-file.npy")
    assert Path(filename).exists()
    assert Path(filename).is_file()
    assert Path(filename).suffix == ".npy"

    video = om.load_video(filename)
    assert video.shape == (2, 4, 6)
    assert video.dtype == np.uint8
    assert np.all(video[1] == 1)

    om.print_properties(video)
