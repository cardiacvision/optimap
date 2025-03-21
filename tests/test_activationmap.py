import numpy as np
import pytest

import optimap as om


def test_activationmap():
    video = np.zeros((100, 100, 100), dtype=np.float32)
    video[10, 0, 0] = 1
    for i in range(1, 100):
        video[60, i, i] = 1
    amap = om.activation.compute_activation_map(video, show=False)
    assert amap[0, 0] == 0
    for i in range(1, 100):
        assert amap[i, i] == pytest.approx(50)
    amap = om.activation.compute_activation_map(video, normalize_time=False, show=False)
    assert amap[0, 0] == 10
    for i in range(1, 100):
        assert amap[i, i] == pytest.approx(60)
    for i in range(0, 100):
        amap[i,i] = np.nan
    assert np.all(np.isnan(amap))

def test_find_activations():
    trace = np.zeros(100, dtype=np.float32)
    activations = om.activation.find_activations(trace, show=False)
    assert activations.size == 0

    trace[50] = 1
    activations = om.activation.find_activations(trace, min_duration=2, show=False)
    assert activations.size == 0

    trace[50:55] = 1
    trace[75:80] = 1
    activations = om.activation.find_activations(trace, min_duration=2, threshold=0.9, show=False)
    assert activations.size == 2
    assert activations[0] == 50
    assert activations[1] == 75

    trace = np.ones(100, dtype=np.float32)
    trace[50:55] = 0
    activations = om.activation.find_activations(trace, falling_edge=True, threshold=0.1, show=False)
    assert activations.size == 1
    assert activations[0] == 50


# #%%

# import optimap as om
# filename = "/Users/janl/work/software/optimap/optimap/docs/tutorials/optimap_example_data/mouse_41_120ms_control_iDS.mat"
# video = om.load_video(filename)
# video_filtered = om.video.smooth_spatiotemporal(video, sigma_temporal=1, sigma_spatial=1)
# video_filtered = om.video.mean_filter(video_filtered, size_spatial=5)

# # Normalize the video using a pixelwise sliding window
# video_norm = om.video.normalize_pixelwise_slidingwindow(video_filtered, window_size=200)
# mask = om.background_mask(video[0], show=False)
# mask = om.image.dilate_mask(mask, iterations=5, show=False)
# video_norm[:, mask] = np.nan
# #%%
# import numpy as np
# def bandpass_filter(video: np.ndarray, low_cut: float, high_cut: float, framerate: float) -> np.ndarray:
#     """Bandpass filter a video in the temporal frequency domain.

#     This function applies a bandpass filter along the time axis by performing a real FFT,
#     zeroing out frequency components outside the desired band [low_cut, high_cut] (in Hz),
#     and then reconstructing the filtered signal with an inverse FFT.

#     Parameters
#     ----------
#     video : {t, x, y} ndarray
#         Video to filter.
#     low_cut : float
#         Lower cutoff frequency in Hz.
#     high_cut : float
#         Upper cutoff frequency in Hz.
#     framerate : float
#         Video frame rate in Hz.

#     Returns
#     -------
#     {t, x, y} ndarray
#         Bandpass filtered video.
#     """
#     if video.ndim != 3:
#         raise ValueError("video has to be 3 dimensional")
    
#     t = video.shape[0]
#     # Perform FFT along time axis
#     fft_video = np.fft.rfft(video, axis=0)
#     freqs = np.fft.rfftfreq(t, d=1.0/framerate)
    
#     # Create bandpass mask (filter frequencies outside the [low_cut, high_cut] band)
#     mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
    
#     # Apply mask (broadcast mask to each spatial pixel)
#     fft_video_filtered = fft_video * mask[:, None, None]
    
#     # Reconstruct the video by applying the inverse FFT
#     filtered_video = np.fft.irfft(fft_video_filtered, n=t, axis=0)
#     return filtered_video
# video2 = video.copy().astype(np.float32)
# video2[:, mask] = np.nan
# filtered = bandpass_filter(video2, low_cut=1, high_cut=200, framerate=1000)
# #%%
# import monochrome as mc
# mc.show(filtered)
# #%%
# mc.show(video)
# #%%
# video_norm_mean = om.video.mean_filter(video_norm, size_spatial=5)
# #%%
# amap = compute_activation_map(video_norm_mean[12:27], inverted=True, fractions=True, show_contours=True)
# # %%
# find_activations(1 - video_norm, min_duration=10)
# #%%
# list(range(10, 11))
# #%%