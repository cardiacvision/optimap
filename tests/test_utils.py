import logging
from pathlib import Path

import numpy as np
import pooch
import pytest

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


@pytest.mark.parametrize("level", [logging.NOTSET, logging.INFO])
def test_download_silent_restores_log_level(monkeypatch, level):
    logger = pooch.get_logger()
    original_level = logger.level
    logger.setLevel(level)
    calls = {}

    def retrieve(**kwargs):
        calls.update(kwargs, level=logger.level)
        return "file.npy"

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    try:
        om.utils.download_example_data("test-download-file.npy", silent=True)
        assert calls["level"] == logging.WARNING
        assert calls["progressbar"] is False  # used to be True if the previous log level was NOTSET
        assert logger.level == level
    finally:
        logger.setLevel(original_level)
