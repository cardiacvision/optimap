from pathlib import Path

import optimap as om
import numpy as np

def test_retrieve_sample():
    filename = om.utils.retrieve_example_data('optimap-test-download-file.npy')
    assert Path(filename).exists()
    assert Path(filename).is_file()
    assert Path(filename).suffix == '.npy'

    video = om.load_video(filename)
    assert video.shape == (2, 4, 6)
    assert video.dtype == np.uint8
    assert np.all(video[1] == 1)