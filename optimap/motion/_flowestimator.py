import warnings
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm


class FlowEstimator:
    """Optical flow estimator class which wraps OpenCV's optical flow methods.
    Supports CPU and GPU methods (if CUDA is available). See :ref:`opencv` for how to install OpenCV with CUDA support.

    See :cite:p:`Lebert2022` for a comparison and discussion of the different optical flow methods for optical mapping
    data. We recommend and default to the Farnebäck method.

    List of supported optical flow methods:

    * ``'farneback_cpu'`` (CPU) or ``'farneback'`` (GPU): :cite:p:`Farneback2003`
    * ``'brox'`` (GPU): :cite:p:`Brox2004`
    * ``'lk'`` (GPU): pyramidal Lucas-Kanade optical flow :cite:p:`Bouguet1999`
    * ``'tvl1_cpu'`` (CPU) or ``'tvl1'`` (GPU): :cite:p:`Zach2007`
    * ``'nvidia'``: NVIDIA Optical Flow SDK 1.0 (GPU)
    * ``'nvidia2'``: NVIDIA Optical Flow SDK 2.0 (GPU)

    The functions :py:func:`estimate_displacements` and :py:func:`estimate_reverse_displacements` functions are wrappers
    around this class for convenience.

    .. warning:: This class expects images with values in the range [0,1]. Except for the ``'brox'`` method, all images are internally converted to uint8 before calculating the optical flow. This is because the OpenCV CUDA optical flow methods only support uint8 images. This may lead to unexpected results if the images are not in the range [0,1].
    """  # noqa: E501

    #: parameters for Farneback optical flow
    farneback_params = {}
    #: parameters for TVL1 optical flow
    tvl1_params = {}
    #: parameters for LK optical flow
    lk_params = {}
    #: parameters for Brox optical flow
    brox_params = {}

    def __init__(self):
        self.cuda_supported = cv2.cuda.getCudaEnabledDeviceCount() > 0  #: whether CUDA is supported
        if self.cuda_supported:
            self.default_method = "farneback"  #: default optical flow method
        else:
            self.default_method = "farneback_cpu"

    def _get_cv2_estimator(self, method: str, img_shape=None):
        if method == "farneback":
            return cv2.cuda_FarnebackOpticalFlow.create(**self.farneback_params)
        elif method == "farneback_cpu":
            return cv2.FarnebackOpticalFlow.create(**self.farneback_params)
        elif method == "tvl1":
            return cv2.cuda_OpticalFlowDual_TVL1.create(**self.tvl1_params)
        elif method == "tvl1_cpu":
            return cv2.optflow_DualTVL1OpticalFlow.create(**self.tvl1_params)
        elif method == "lk":
            return cv2.cuda_DensePyrLKOpticalFlow.create(**self.lk_params)
        elif method == "brox":
            return cv2.cuda_BroxOpticalFlow.create(**self.brox_params)
        elif method == "nvidia":
            if img_shape is None:
                msg = "shape must be specified for nvidia optical flow method"
                raise ValueError(msg)
            return cv2.cuda_NvidiaOpticalFlow_1_0.create(
                img_shape,
                cv2.cuda.NVIDIA_OPTICAL_FLOW_1_0_NV_OF_PERF_LEVEL_SLOW,
                enableTemporalHints=False,
                enableExternalHints=False,
                enableCostBuffer=False,
                gpuId=0,
            )
        elif method == "nvidia2":
            if img_shape is None:
                msg = "shape must be specified for nvidia optical flow method"
                raise ValueError(msg)
            return cv2.cuda_NvidiaOpticalFlow_2_0.create(
                img_shape,
                cv2.cuda.NVIDIA_OPTICAL_FLOW_2_0_NV_OF_PERF_LEVEL_SLOW,
                enableTemporalHints=False,
                enableExternalHints=False,
                enableCostBuffer=False,
                gpuId=0,
            )
        else:
            msg = f"Unknown optical flow method specified: {method}"
            raise NotImplementedError(msg)

    @staticmethod
    def _upscale_imgs(imgs: Union[np.ndarray, list]):
        # allow both list of images and single images as input
        if isinstance(imgs, np.ndarray) and imgs.ndim == 2:
            imgs = [imgs]
            single_image = True
        else:
            single_image = False

        upsampled = []
        for img in imgs:
            img = cv2.resize(img, (img.shape[0] * 2, img.shape[1] * 2), cv2.INTER_CUBIC)
            upsampled.append(img)

        if single_image:
            return upsampled[0]
        else:
            return np.array(upsampled)

    @staticmethod
    def _downscale_flow(flows: Union[np.ndarray, list]):
        downscaled = []
        for flow in flows:
            flow = (
                cv2.resize(flow, (int(flow.shape[1] * 0.5), int(flow.shape[0] * 0.5)))
                * 0.5
            )
            downscaled.append(flow)
        return np.array(downscaled)

    @staticmethod
    def _as_uint8(imgs: Union[np.ndarray, list]):
        if isinstance(imgs, list):
            imgs = np.array(imgs)
        return (imgs * 255).astype(np.uint8)

    @staticmethod
    def _check_and_convert(imgs: Union[np.ndarray, list], method: str):
        if method == "brox":
            return imgs

        if isinstance(imgs, list):
            imgs = np.array(imgs)
        if imgs.min() < 0 or imgs.max() > 1:
            warnings.warn(
                "WARNING: image values are not in range [0,1], which may lead to unexpected motion tracking"
            )
        return (imgs * 255).astype(np.uint8)

    def _estimate(
        self,
        vid1,
        vid2,
        method: str = None,
        show_progress: bool = True,
        upscale_imgs: bool = False,
    ):
        if method is None:
            method = self.default_method

        if len(vid1) != len(vid2):
            msg = f"Error: arrays have unequal length: {len(vid1)=} != {len(vid2)=}"
            raise ValueError(msg)
        if vid1[0].shape != vid2[0].shape:
            msg = f"Error: images have different dimensions {vid1[0].shape=} != {vid2[0].shape=}"
            raise ValueError(msg)

        Nt = len(vid1)
        img_shape = vid1[0].shape
        flows = np.zeros((Nt, *img_shape, 2), dtype=np.float32)
        estimator = self._get_cv2_estimator(method, img_shape)
        uses_gpu = method not in ["farneback_cpu", "tvl1_cpu"]
        description = "calculating flows"
        if uses_gpu:
            description += " (GPU)"
        else:
            description += " (CPU)"

        if method.startswith("nvidia") and img_shape[0] * img_shape[1] < 160 * 160:
            # NVDIA optical flow SDK has an undocumented minimum image size (approx 160x160 px),
            # so we need to upscale the images and downscale the results if necessary
            upscale_imgs = True

        if upscale_imgs:
            vid1 = self._upscale_imgs(vid1)
            vid2 = self._upscale_imgs(vid2)

        vid1 = self._check_and_convert(vid1, method)
        vid2 = self._check_and_convert(vid2, method)

        if method in ["farneback_cpu", "tvl1_cpu", "nvidia"]:
            # Methods which don't need GpuMat, note that `nvidia` still runs on GPU
            for i, (img1, img2) in enumerate(
                tqdm(
                    zip(vid1, vid2),
                    desc=description,
                    disable=not show_progress,
                    total=Nt,
                )
            ):
                flow = estimator.calc(img1, img2, None)
                if method == "nvidia":
                    flow = estimator.upSampler(
                        flow,
                        (img1.shape[1], img1.shape[0]),
                        estimator.getGridSize(),
                        None,
                    )
                if upscale_imgs:
                    # downscale flow to original size
                    flow = (
                        cv2.resize(flow, img_shape, interpolation=cv2.INTER_AREA) * 0.5
                    )
                flows[i] = flow

            if method == "nvidia":
                estimator.collectGarbage()
        else:
            img1_gpu = cv2.cuda_GpuMat()
            img2_gpu = cv2.cuda_GpuMat()
            for i, (img1, img2) in enumerate(
                tqdm(
                    zip(vid1, vid2),
                    desc=description,
                    disable=not show_progress,
                    total=Nt,
                )
            ):
                img1_gpu.upload(img1)
                img2_gpu.upload(img2)
                flow_gpu = estimator.calc(img1_gpu, img2_gpu, None)
                flow = flow_gpu.download()
                if upscale_imgs:
                    # downscale flow to original size
                    flow = (
                        cv2.resize(flow, img_shape, interpolation=cv2.INTER_AREA) * 0.5
                    )
                flows[i] = flow
        return flows

    def estimate(
        self,
        imgs: Union[np.ndarray, list],
        ref_img: np.ndarray,
        method: str = None,
        show_progress: bool = True,
    ):
        """Estimate optical flow between every frame of a video and a reference frame. The returned optical flow
        is an array of 2D flow fields which can be used to warp the video to the reference frame (motion compensation).

        Parameters
        ----------
        imgs : Union[np.ndarray, list]
            Video to estimate optical flow for (list of grayscale images or 3D array {t, x, y})
        ref_img : np.ndarray
            Reference image to estimate optical flow to (grayscale image {x, y})
        method : str, optional
            Optical flow method to use, by default ``None`` which means ``'farneback'`` if a CUDA GPU is
            available, or ``'farneback_cpu'`` otherwise
        show_progress : bool, optional
            show progress bar, by default True

        Returns
        -------
        np.ndarray
            optical flow array of shape {t, x, y, 2}
        """
        vid1 = np.repeat(ref_img[np.newaxis, :, :], len(imgs), axis=0)
        vid2 = imgs
        return self._estimate(vid1, vid2, method, show_progress)

    def estimate_reverse(
        self,
        imgs: Union[np.ndarray, list],
        ref_img: np.ndarray,
        method: str = None,
        show_progress: bool = True,
    ):
        """Estimate optical flow between a reference frame and every frame of a video. The returned optical flow
        is an array of 2D flow fields which can be used to warp the video to the reference frame (motion compensation).

        Parameters
        ----------
        imgs : Union[np.ndarray, list]
            Video to estimate optical flow for (list of grayscale images or 3D array {t, x, y})
        ref_img : np.ndarray
            Reference image to estimate optical flow to (grayscale image {x, y})
        method : str, optional
            Optical flow method to use, by default ``None`` which means ``'farneback'`` if a CUDA GPU is
            available, or ``'farneback_cpu'`` otherwise
        show_progress : bool, optional
            show progress bar, by default True

        Returns
        -------
        np.ndarray
            optical flow array of shape {t, x, y, 2}
        """
        vid1 = imgs
        vid2 = np.repeat(ref_img[np.newaxis, :, :], len(imgs), axis=0)
        return self._estimate(vid1, vid2, method, show_progress)

    def estimate_imgpairs(
        self, img_pairs, method: str = None, show_progress: bool = True
    ):
        """Estimate optical flow between list of two images.

        Parameters
        ----------
        img_pairs : list of tuples of np.ndarray
            List of grayscale image pairs to estimate optical flow for
        method : str, optional
            Optical flow method to use, by default None
        show_progress : bool, optional
            show progress bar, by default True

        Returns
        -------
        np.ndarray
            optical flow array of shape {N, x, y, 2}
        """
        imgs1 = [img1 for img1, img2 in img_pairs]
        imgs2 = [img2 for img1, img2 in img_pairs]
        return self._estimate(imgs1, imgs2, method, show_progress)
