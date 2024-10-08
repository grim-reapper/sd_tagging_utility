# Install necessary packages
#!pip install diffusers gradio
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import gradio as gr

# Load the model and scheduler
model_id = "google/ddpm-celebahq-256"
ddpm = DDPMPipeline.from_pretrained(model_id)

def generate_image():
    # Run the pipeline in inference (sample random noise and denoise)
    image = ddpm()["sample"]
    return image[0]

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[],
    outputs="image",
    title="DDPM Image Generator",
    description="Generate images using the DDPM (Denoising Diffusion Probabilistic Model) from the CelebA-HQ dataset."
)

# Launch the Gradio app
demo.launch()