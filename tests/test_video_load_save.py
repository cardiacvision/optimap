import struct
from pathlib import Path

import numpy as np
import pytest

import optimap as om


def test_save_tiff_suffix(tmpdir):
    vid = np.zeros((2, 4, 4), dtype=np.uint16)
    om.video.save_tiff(Path(tmpdir) / "video.tif", vid)
    assert (Path(tmpdir) / "video.tif").exists()
    assert not (Path(tmpdir) / "video.tiff").exists()


def test_save_tiff_folder_format(tmpdir):
    folder = Path(tmpdir) / "frames"
    om.video.save_tiff_folder(folder, np.zeros((2, 4, 4), dtype=np.uint16))
    assert sorted(f.name for f in folder.iterdir()) == ["frame_000.tiff", "frame_001.tiff"]


def test_matlab_fieldname(tmpdir):
    import scipy.io

    fn = Path(tmpdir) / "multi.mat"
    scipy.io.savemat(fn, {"a": np.zeros((3, 4, 4)), "b": np.ones((5, 6, 6)), "s": "text"})
    assert om.load_video(fn, fieldname="a").shape == (3, 4, 4)
    assert om.load_video(fn, fieldname="b").shape == (5, 6, 6)
    with pytest.raises(ValueError):
        om.load_video(fn, fieldname="missing")
    with pytest.raises(ValueError):
        om.load_video(fn, fieldname="s")


def _write_multirecorder(filename, video, framerate=500.0, comment="test"):
    nt, nx, ny = video.shape
    with open(filename, "wb") as f:
        f.write(b"f")
        f.write(struct.pack("<iiii", 0, nt, ny, nx))
        f.write(b"\0" * 8)
        f.write(struct.pack("<i", int(framerate * 100000)))
        f.write(b"2023-06-22 10:11:12:123".ljust(24, b"\0"))
        f.write(comment.encode().ljust(971, b"\0"))
        for t, frame in enumerate(video.astype("<u2")):
            f.write(frame.tobytes())
            f.write(struct.pack("<Q", t))  # frame time


def test_multirecorder_importer(tmpdir):
    video = np.arange(3 * 4 * 6, dtype=np.uint16).reshape(3, 4, 6)
    fn = Path(tmpdir) / "video.dat"
    _write_multirecorder(fn, video)

    assert np.array_equal(om.load_video(fn), video)
    assert np.array_equal(om.load_video(fn, start_frame=1, frames=1), video[1:2])
    assert om.load_metadata(fn)["framerate"] == 500.0
    with pytest.raises(ValueError):
        om.load_video(fn, start_frame=3)


def test_multirecorder_metadata_not_shared(tmpdir):
    from optimap.video._importers import MultiRecorderImporter

    fn1, fn2 = Path(tmpdir) / "a.dat", Path(tmpdir) / "b.dat"
    _write_multirecorder(fn1, np.zeros((3, 4, 6)), framerate=100, comment="first")
    _write_multirecorder(fn2, np.zeros((5, 4, 6)), framerate=200, comment="second")

    first, second = MultiRecorderImporter(fn1), MultiRecorderImporter(fn2)
    assert first.get_metadata()["Number of frames"] == 3
    assert first.get_metadata()["comment"] == "first"
    assert second.get_metadata()["Number of frames"] == 5
    assert second.get_metadata()["comment"] == "second"


def _write_micam_ultima(folder, name, background, blocks, extra_meta=""):
    """Writes a synthetic MiCAM ULTIMA recording, `blocks` is a list of arrays of shape (blk, Nx, Ny)."""
    nx, ny = background.shape
    blk = len(blocks[0])
    background.astype("<u2").tofile(folder / f"{name}.rsm")
    files = [f"{name}.rsm"]
    for i, block in enumerate(blocks):
        block.astype("<i2").tofile(folder / f"{name}_{i}.rsd")
        files.append(f"{name}_{i}.rsd")
    frames = blk * len(blocks)
    header = [
        "header",
        f"/x={ny}/y={nx}/lskp=0/rskp=0/blk={blk}/",
        "",
        f"page_frames={frames}",
        extra_meta,
        "Data-File-List",
        *files,
    ]
    (folder / f"{name}.rsh").write_text("\n".join(header))
    return folder / f"{name}.rsh"


def test_micam_ultima_importer(tmpdir):
    from optimap.video._importers import MiCAM_ULTIMA_Importer

    folder = Path(tmpdir)
    background = np.full((4, 6), 100, dtype=np.uint16)
    blocks = [np.full((2, 4, 6), i, dtype=np.int16) for i in range(3)]
    fn = _write_micam_ultima(folder, "a", background, blocks, extra_meta="only_in_a=1")
    fn2 = _write_micam_ultima(folder, "b", background, blocks[:1])

    video = om.load_video(fn)
    assert video.shape == (6, 4, 6)
    assert video.dtype == np.uint16
    assert video[:, 0, 0].tolist() == [100, 100, 101, 101, 102, 102]

    first, second = MiCAM_ULTIMA_Importer(fn), MiCAM_ULTIMA_Importer(fn2)
    assert first.get_metadata()["page_frames"] == "6"
    assert second.get_metadata()["page_frames"] == "2"
    assert "only_in_a" not in second.get_metadata()
