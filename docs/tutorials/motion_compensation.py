# %% [markdown]
# # Tutorial 2: Motion Compensation
#
# This tutorial focuses on the motion estimation and compensation capabilities of the `optimap` library.

# %%
import optimap as om
import monochrome as mc  # remove this if you don't have monochrome installed

filename = om.utils.retrieve_example_data('Example_01_Sinus_Rabbit_Basler_acA720-520um.npy')
video = om.load_video(filename)
video = om.video.rotate_left(video)

# %% [markdown]
#

# %%
om.video.play(video, title="original video with strong deformation");

# %%
warped = om.motion.motion_compensate(video, 5, ref_frame=40)
flows_nocontrast = om.motion.estimate_displacements(video, 40)
warped_nocontrast = om.motion.warp_video(video, flows_nocontrast)
om.video.playn([video, warped, warped_nocontrast],
               titles=["original video", "with contrast-enhancement", "w/o contrast-enhancement"], figsize=(8, 3.5));

# %%
warped_ref0 = om.motion_compensate(video, contrast_kernel=5, ref_frame=0)
warped_ref40 = om.motion_compensate(video, contrast_kernel=5, ref_frame=40)
om.video.playn([video, warped_ref40, warped_ref0], titles=["original video", "compensated ref 40", "compensated ref 0"], figsize=(8, 3.5));

# %%
contrast3 = om.motion.contrast_enhancement(video[:300], 3)
contrast5 = om.motion.contrast_enhancement(video[:300], 5)
contrast9 = om.motion.contrast_enhancement(video[:300], 9)
om.video.playn([contrast3, contrast5, contrast9],
               titles=["contrast kernel 3", "contrast kernel 5", "contrast kernel 9"],
               skip_frame=3,
               figsize=(8, 3.5));
