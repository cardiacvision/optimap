from pathlib import Path

import numpy as np

import optimap as om


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


def test_mask_export(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    mask = img > 0.5

    fn = Path(tmpdir) / "test.npy"
    om.image.save_mask(mask, fn)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.png"
    om.image.save_mask(mask, fn)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn.unlink()
    om.image.save_mask(mask, fn, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.tiff"
    om.image.save_mask(mask, fn)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn.unlink()
    om.image.save_mask(mask, fn, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)


def test_16bit_mask_save(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    mask = img > 0.5
    img = (img * 65535).astype(np.uint16)

    fn = Path(tmpdir) / "test.png"
    om.image.save_mask(mask, fn, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.tiff"
    om.image.save_mask(mask, fn, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)
