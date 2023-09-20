import optimap as om
import numpy as np
from pathlib import Path

def test_image_export(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)

    fn = tmpdir / "test.npy"
    om.image.save_image(img, Path(fn))
    img2 = om.image.load_image(Path(fn))
    assert np.allclose(img, img2)

def test_16bit_export(tmpdir):
    """Test if 16-bit PNG and TIFF export works."""
    img = np.random.rand(100, 100).astype(np.float32)
    img = (img * 65535).astype(np.uint16)

    fn = tmpdir / "test.png"
    om.image.save_image(img, fn)
    img2 = om.image.load_image(fn)
    assert img.dtype == img2.dtype
    assert np.allclose(img, img2)

    fn = tmpdir / "test.tiff"
    om.image.save_image(img, fn)
    img2 = om.image.load_image(fn)
    assert img.dtype == img2.dtype
    assert np.allclose(img, img2)
