import optimap as om
import numpy as np
from tempfile import TemporaryDirectory

def test_image_export():
    img = np.random.rand(100, 100).astype(np.float32)
    with TemporaryDirectory() as tmpdir:
        fn = tmpdir + "/test.npy"
        om.image.save_image(img, fn)
        img2 = om.image.load_image(fn)
        assert np.allclose(img, img2)

    # Test if 16-bit PNG and TIFF export works
    with TemporaryDirectory() as tmpdir:
        img = (img * 65535).astype(np.uint16)
        fn = tmpdir + "/test.png"
        om.image.save_image(img, fn)
        img2 = om.image.load_image(fn)
        assert img.dtype == img2.dtype
        assert np.allclose(img, img2)

    with TemporaryDirectory() as tmpdir:
        img = (img * 65535).astype(np.uint16)
        fn = tmpdir + "/test.tiff"
        om.image.save_image(img, fn)
        img2 = om.image.load_image(fn)
        assert img.dtype == img2.dtype
        assert np.allclose(img, img2)
