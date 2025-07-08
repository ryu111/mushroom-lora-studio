from diffusers import StableDiffusionPipeline
import torch

class CounterfeitV30Model:
    def load_pipeline(self):
        return StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("mps")