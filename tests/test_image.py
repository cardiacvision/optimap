from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest

import optimap as om


def test_image_save(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)

    fn = tmpdir / "test.npy"
    om.image.save_image(fn, img)
    img2 = om.image.load_image(fn)
    assert np.allclose(img, img2)

def test_16bit_save(tmpdir):
    """Test if 16-bit PNG and TIFF save works."""
    img = np.random.rand(100, 100).astype(np.float32)
    img = (img * 65535).astype(np.uint16)

    fn = tmpdir / "test.png"
    om.image.save_image(fn, img)
    img2 = om.image.load_image(fn)
    assert img.dtype == img2.dtype
    assert np.allclose(img, img2)

    fn = tmpdir / "test.tiff"
    om.image.save_image(fn, img)
    img2 = om.image.load_image(fn)
    assert img.dtype == img2.dtype
    assert np.allclose(img, img2)


def test_compat_save(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    img = (img * 65535).astype(np.uint16)

    fn = tmpdir / "test.png"
    om.image.save_image(fn, img, compat=True)
    img2 = om.image.load_image(fn)
    assert img2.dtype == np.dtype("uint8")
    assert img2.max() > 100


def test_image_export(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    
    fn = tmpdir / "test.png"
    om.image.export_image(fn, img)
    img2 = om.image.load_image(fn)
    assert img2.ndim == 3
    assert img2.dtype == np.dtype("uint8")

    fn = tmpdir / "test.jpg"
    om.image.export_image(fn, img, vmin=0.1, vmax=0.9, cmap=plt.get_cmap("viridis"))

    fn = tmpdir / "test.tiff"
    om.image.export_image(fn, img, vmin=0.1, vmax=0.9, cmap="viridis")
    img2 = om.image.load_image(fn)
    assert img2.ndim == 3
    assert img2.dtype == np.dtype("uint8")


def test_mask_save(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    mask = img > 0.5

    fn = Path(tmpdir) / "test.npy"
    om.image.save_mask(fn, mask)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.png"
    om.image.save_mask(fn, mask)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn.unlink()
    om.image.save_mask(fn, mask, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.tiff"
    om.image.save_mask(fn, mask)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn.unlink()
    om.image.save_mask(fn, mask, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)


def test_16bit_mask_save(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    mask = img > 0.5
    img = (img * 65535).astype(np.uint16)

    fn = Path(tmpdir) / "test.png"
    om.image.save_mask(fn, mask, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)

    fn = Path(tmpdir) / "test.tiff"
    om.image.save_mask(fn, mask, img)
    mask2 = om.image.load_mask(fn)
    assert fn.exists()
    assert mask2.dtype == bool
    assert np.allclose(mask, mask2)


def test_save_arguement_order(tmpdir):
    img = np.random.rand(100, 100).astype(np.float32)
    mask = img > 0.5

    fn_img = tmpdir / "test.npy"
    fn_mask = tmpdir / "test.png"
    with pytest.warns(DeprecationWarning):
        om.image.save_image(img, fn_img)
    img2 = om.image.load_image(fn_img)
    assert np.allclose(img, img2)

    with pytest.warns(DeprecationWarning):
        om.image.save_image(img, str(fn_img))
    img2 = om.image.load_image(fn_img)
    assert np.allclose(img, img2)

    with pytest.warns(DeprecationWarning):    
        om.image.save_mask(mask, fn_mask)
    mask2 = om.image.load_mask(fn_mask)
    assert np.allclose(mask, mask2)

    with pytest.warns(DeprecationWarning):
        om.image.save_mask(mask, str(fn_mask))
    mask2 = om.image.load_mask(fn_mask)
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


def test_collage():
    img = np.random.rand(100, 100).astype(np.float32)
    img_rgba = np.random.rand(100, 100, 4).astype(np.float32)

    img2 = om.image.collage([img, img, img, img], ncols=2)
    assert img2.shape == (200, 200)
    assert img2.dtype == np.float32

    img2 = om.image.collage([img_rgba, img_rgba, img_rgba, img_rgba], ncols=4)
    assert img2.shape == (100, 400, 4)
    assert img2.dtype == np.float32

    img2 = om.image.collage([img, img, img, img], ncols=1, padding=10)
    assert img2.shape == (430, 100)
    assert img2.dtype == np.float32

    img2 = om.image.collage([img, img, img, img], ncols=2, padding=10)
    assert img2.shape == (210, 210)
    assert img2.dtype == np.float32