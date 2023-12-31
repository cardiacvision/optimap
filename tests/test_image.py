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


def test_rotation_and_flip():
    img = np.random.rand(100, 100).astype(np.float32)
    img_rgba = np.random.rand(100, 100, 4).astype(np.float32)

    def rotations(img):
        img2 = om.image.rotate_left(img)
        img2 = om.image.rotate_right(img2)
        assert np.allclose(img, img2)
        img2 = om.image.flip_up_down(img2)
        img2 = om.image.flip_up_down(img2)
        assert np.allclose(img, img2)
        img2 = om.image.flip_left_right(img2)
        img2 = om.image.flip_left_right(img2)
        assert np.allclose(img, img2)

    rotations(img)
    rotations(img_rgba)

def test_crop_pad():
    img = np.random.rand(100, 100).astype(np.float32)
    img_rgba = np.random.rand(100, 100, 4).astype(np.float32)

    img2 = om.image.crop(img, 10)
    assert img2.shape == (80, 80)
    assert np.allclose(img[10:-10, 10:-10], img2)
    img2 = om.image.pad(img2, 10)
    assert img2.shape == (100, 100)
    assert np.allclose(img[10:-10, 10:-10], img2[10:-10, 10:-10])

    img2 = om.image.crop(img_rgba, 10)
    assert img2.shape == (80, 80, 4)
    assert np.allclose(img_rgba[10:-10, 10:-10], img2)
    img2 = om.image.pad(img2, 10)
    assert img2.shape == (100, 100, 4)
    assert np.allclose(img_rgba[10:-10, 10:-10], img2[10:-10, 10:-10])

    assert np.allclose(img, om.image.pad(img, 0))
    assert np.allclose(img, om.image.crop(img, 0))


def test_resize():
    img = np.random.rand(100, 100).astype(np.float32)
    img_rgba = np.random.rand(100, 100, 4).astype(np.float32)

    img2 = om.image.resize(img, (50, 50))
    assert img2.shape == (50, 50)
    assert img2.dtype == np.float32

    img2 = om.image.resize(img_rgba, (50, 50))
    assert img2.shape == (50, 50, 4)
    assert img2.dtype == np.float32

    img2 = om.image.resize(img, (50, 50), interpolation="nearest")
    assert img2.shape == (50, 50)
