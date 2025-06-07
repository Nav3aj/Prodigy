"""generate_images.py
Instruction: Generate an image from a text prompt using Stable Diffusion.
"""

# Topic: Imports
import argparse
from diffusers import StableDiffusionPipeline
import torch

# Topic: Main Function
def main(prompt: str, output: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    image = pipe(prompt).images[0]
    image.save(output)
    print(f"Saved -> {output}")

# Topic: CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image gen with Stable Diffusion")
    p.add_argument("--prompt",  required=True, help="Text prompt")
    p.add_argument("--output", default="output.png", help="Output file")
    args = p.parse_args()
    main(args.prompt, args.output)
