# %% [markdown]
# # Tutorial 1: Basics
#
# This tutorial will walk you through the basics of using the `optimap` package.
#
# First, let's import optimap and the other packages we will need:
#

# %%
import optimap as om
import monochrome as mc
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Loading a video file
#
# We now have access to all the functions in the `optimap` package. Let's start by loading a video file. We will use the `optimap.video.load` function to load a sample video file.
#
# The recording of a beating Rabbit heart stained with the voltage-sensitive dye di-4-ANEPPS was acquired at 500 fps using a Basler acA720-520um camera. The action potentials are inverted, i.e. an upstroke is observed as a negative deflection.
#
# We have extracted a short section of the original recording and saved the raw data as a numpy file (`.npy`). See {func}`optimap.video.load` for a list of supported file formats.
#
# * experimenter: Jan Lebert, Shrey Chowdhary & Jan Christoph
# * institution: University of California, San Francisco, USA

# %%
filepath = om.utils.retrieve_example_data('Example_02_VF_Rabbit_Di-4-ANEPPS_Basler_acA720-520um.npy')
video = om.load_video(filepath)

om.print_properties(video)

# %% [markdown]
# optimap imports videos as numpy arrays with the shape (Time, Height, Width). This convention is used throughout the library.
#
# ## Playing videos
# Videos can be viewed either with the builtin matplotlib viewer:

# %%
om.video.play(video);

# %% [markdown]
# or with the Monochrome viewer, which is a separate project.
#
# In Monochrome click on the video to view time traces at the selected positions.

# %%
mc.show(video, "raw video")

# %% [markdown]
# ## Viewing and extracting traces
#
# Time traces can be viewed and extracted interactively using the {func}`optimap.trace.select_traces` function. Click on the image to select positions, right click to remove positions. Close the window to continue.

# %%
traces, positions = om.trace.select_traces(video, size=3)

# %% [markdown]
# The `size` parameter controls the dimensions of the window surrounding the chosen location, from which the average is computed. By default, this window is a rectangle with dimensions `(size, size)`.
#
# To get the exact pixel values without averaging, set `size=1`. If you'd like to display the time axis in seconds rather than frames, use the `fps` parameter.

# %%
traces = om.extract_traces(video, positions, size=1, show=True, fps=500)

# %% [markdown]
# The `window` parameter can be used to define the window function, `'disc'` uses a circular region with radius `size` around the position. See {func}`optimap.trace.select_traces` for more information.
#
# Internally {func}`optimap.trace.extract_traces` uses {func}`optimap.trace.show_traces` to plot traces. In general, all plotting functions in optimap have an `ax` parameter which can be used to specify a custom matplotlib axes object.
#
# For example, we can create a figure with two subplots and show the positions on the first subplot and the traces on the second subplot with milliseconds as time unit:

# %%
fig, axs = plt.subplots(1,2, figsize=(10,5))

om.trace.show_positions(video[0], positions, ax=axs[0])

x_axis_ms = (np.arange(video.shape[0]) / 500.0) * 1000
traces = om.extract_traces(video[:300],
                           positions,
                           x=x_axis_ms[:300],
                           size=5,
                           window='disc',
                           ax=axs[1],
                           show=True)
axs[1].set_xlabel('Time [ms]')
plt.show()

# %% [markdown]
# ## Motion Compensation
#
# The heart is beating and slightly moving during the recording. Even though the motion is small, it can have a strong effect on the time traces in the form of motion artifacts. We can use the {func}`optimap.motion.motion_compensate` function to compensate for the motion using the steps described in {cite}`Christoph2018a` and {cite}`Lebert2022`. See [](motion_compensation) for detailed information and examples.

# %%
warped = om.motion_compensate(video,
                              contrast_kernel=5,
                              presmooth_spatial=1,
                              presmooth_temporal=1,)

# %% [markdown]
# Let's view the original video and motion-compensated video side by side using {func}`optimap.video.play2`:

# %%
om.video.play2(video,
               warped,
               title1="with motion",
               title2="without motion",
               skip_frame=3);

# %% [markdown]
# Let's save the motion-compensated recording as a tiff stack and also render it to a .mp4 video file:

# %%
om.video.save_video(warped, 'warped_recording.tiff')
om.video.export_video(warped, 'warped_recording.mp4', fps=50)

# %% [markdown]
# ## Fluorescence wave isolation
#
# To better visualize the action potential propagation we can compute a pixel-wise normalization to [0, 1] using a sliding/rolling window of 60 frames.

# %%
norm_raw = om.video.normalize_pixelwise_slidingwindow(video, window_size=60)
norm_warped = om.video.normalize_pixelwise_slidingwindow(warped, window_size=60)
om.video.play2(norm_raw, norm_warped, title1="with motion", title2="without motion");

# %%
mask = om.background_mask(warped[0])
# norm_warped[:, mask] = 0

# %%
