from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
spectral_shifts_ckpt_dir = "weights/v1-5-pruned-emaonly.safetensors"
unet = load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="unet")
text_encoder = load_text_encoder_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="text_encoder")
# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
    text_encoder=text_encoder,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.show()