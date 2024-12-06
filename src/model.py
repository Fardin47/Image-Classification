from diffusers import StableDiffusionPipeline
import torch


def load_pipeline(
    pretrained_model_name="runwayml/stable-diffusion-v1-5", device="cuda"
):
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)
    return pipeline
