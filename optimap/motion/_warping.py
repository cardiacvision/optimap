import cv2
import numpy as np
from tqdm import tqdm


def warp_video(
    video: np.ndarray,
    displacements: np.ndarray,
    show_progress=False,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
):
    """Warps a video according to the given optical flow. Uses GPU if available.

    Parameters
    ----------
    video : {t, x, y} ndarray or list of images
        video to warp
    displacements : {t, x, y, 2} ndarray or list of ndarrays
        optical flow fields
    show_progress : bool, optional
        show progress bar
    interpolation : int, optional
        interpolation method (see cv2.remap), by default ``cv2.INTER_CUBIC``
    borderMode : int, optional
        border mode (see cv2.remap), by default ``cv2.BORDER_REFLECT101``
    borderValue : float, optional
        border value (see cv2.remap), by default 0.0

    Returns
    -------
    np.ndarray
        {t, x, y} ndarray
    """
    f = warp_video_gpu if cv2.cuda.getCudaEnabledDeviceCount() > 0 else warp_video_cpu
    return f(video, displacements, interpolation, borderMode, borderValue, show_progress)


def warp_image(
    img: np.ndarray,
    displacement: np.ndarray,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
):
    """Warps an image according to the given optical flow. Uses GPU if available.

    Parameters
    ----------
    img : {x, y} ndarray
        image to warp
    displacement : {x, y, 2} ndarray
        optical flow field
    show_progress : bool, optional
        show progress bar
    interpolation : int, optional
        interpolation method (see cv2.remap), by default ``cv2.INTER_CUBIC``
    borderMode : int, optional
        border mode (see cv2.remap), by default ``cv2.BORDER_REFLECT101``
    borderValue : float, optional
        border value (see cv2.remap), by default 0.0

    Returns
    -------
    np.ndarray
        {x, y} ndarray
    """
    f = warp_image_gpu if cv2.cuda.getCudaEnabledDeviceCount() > 0 else warp_image_cpu
    return f(img, displacement, interpolation, borderMode, borderValue)


def warp_video_gpu(
    imgs: np.ndarray,
    flows,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
    show_progress=False,
):
    img_gpu = cv2.cuda_GpuMat()
    flow0 = cv2.cuda_GpuMat()
    flow1 = cv2.cuda_GpuMat()
    img_gpu.create(imgs.shape[1], imgs.shape[2], cv2.CV_32FC1)
    flow0.create(imgs.shape[1], imgs.shape[2], cv2.CV_32FC1)
    flow1.create(imgs.shape[1], imgs.shape[2], cv2.CV_32FC1)

    h, w = flows.shape[1:3]
    X = np.arange(w)
    Y = np.arange(h)[:, np.newaxis]

    warped_arr = []
    assert imgs.shape == flows.shape[:-1]
    for frame, flow in tqdm(
        zip(imgs, flows), total=len(imgs), desc="Warping", disable=not show_progress
    ):
        flow = np.copy(flow)
        flow[:, :, 0] += X
        flow[:, :, 1] += Y

        img_gpu.upload(frame)
        flow0.upload(flow[:, :, 0])
        flow1.upload(flow[:, :, 1])
        warped = cv2.cuda.remap(
            img_gpu,
            flow0,
            flow1,
            interpolation,
            borderMode=borderMode,
            borderValue=borderValue,
        ).download()
        warped_arr.append(warped)

    return np.array(warped_arr)


def warp_image_gpu(
    img: np.ndarray,
    flow,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
):
    assert img.shape == flow.shape[:-1]

    img_gpu = cv2.cuda_GpuMat().upload(img)
    flow0 = cv2.cuda_GpuMat()
    flow1 = cv2.cuda_GpuMat()

    h, w = flow.shape[0:2]
    X = np.arange(w)
    Y = np.arange(h)[:, np.newaxis]
    flow = np.copy(flow)
    flow[:, :, 0] += X
    flow[:, :, 1] += Y

    flow0.upload(flow[:, :, 0])
    flow1.upload(flow[:, :, 1])
    return cv2.cuda.remap(
        img_gpu,
        flow0,
        flow1,
        interpolation,
        borderMode=borderMode,
        borderValue=borderValue,
    ).download()


def warp_video_cpu(
    imgs: np.ndarray,
    flows,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
    show_progress=False,
):
    imgs = imgs.copy()
    flows = flows.copy()

    h, w = flows.shape[1:3]
    X = np.arange(w)
    Y = np.arange(h)[:, np.newaxis]

    warped_arr = []
    assert imgs.shape == flows.shape[:-1]
    for frame, flow in tqdm(
        zip(imgs, flows), total=len(imgs), desc="Warping", disable=not show_progress
    ):
        flow = np.copy(flow)
        flow[:, :, 0] += X
        flow[:, :, 1] += Y

        flow0 = flow[:, :, 0].copy()
        flow1 = flow[:, :, 1].copy()
        warped = cv2.remap(
            frame,
            flow0,
            flow1,
            interpolation,
            borderMode=borderMode,
            borderValue=borderValue,
        )
        warped_arr.append(warped)

    return np.array(warped_arr)


def warp_image_cpu(
    img: np.ndarray,
    flow,
    interpolation=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REFLECT101,
    borderValue=0.0,
):
    assert img.shape == flow.shape[:-1]

    h, w = flow.shape[0:2]
    X = np.arange(w)
    Y = np.arange(h)[:, np.newaxis]

    flow = np.copy(flow)
    flow[:, :, 0] += X
    flow[:, :, 1] += Y

    flow0 = flow[:, :, 0].copy()
    flow1 = flow[:, :, 1].copy()
    return cv2.remap(
        img, flow0, flow1, interpolation, borderMode=borderMode, borderValue=borderValue
    )
