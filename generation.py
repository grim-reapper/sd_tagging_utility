# Paths to your files
# sdxl_checkpoint_path = "F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL.safetensors"  # Existing SDXL checkpoint
# vae_path = "F:/foocus/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors"  # VAE model weights (typically in PyTorch .pth format)
# output_path = "F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL_vae.safetensors"  # Path to save the new checkpoint
import torch
import gradio as gr
from safetensors.torch import load_file
import os
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline

# Function to load a model (either base model or LoRA) from a .safetensors file
def load_safetensors_model(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist.")

    # Load the safetensors file
    model_weights = load_file(file_path)
    return model_weights

# Function to merge base model and LoRA model weights
def merge_lora(base_model_weights, lora_weights, alpha=0.5):
    merged_weights = {}

    # Apply LoRA on top of the base model weights
    for key in base_model_weights.keys():
        if key in lora_weights:
            # Merge weights based on alpha
            merged_weights[key] = (1 - alpha) * base_model_weights[key] + alpha * lora_weights[key]
        else:
            # If the weight is not in LoRA, keep the base model weight
            merged_weights[key] = base_model_weights[key]

    return merged_weights

# Function to generate an image using the merged model
def generate_image(checkpoint_path, lora_path, lora_weight, positive_prompt, negative_prompt, width, height, seed, steps, progress=None):
    # Load base model and LoRA weights
    base_model_weights = load_safetensors_model(checkpoint_path)
    lora_weights = load_safetensors_model(lora_path) if lora_path else None

    # Merge LoRA weights if provided
    if lora_weights:
        model_weights = merge_lora(base_model_weights, lora_weights, alpha=lora_weight)
    else:
        model_weights = base_model_weights

    # Use a pretrained pipeline (example: SD 1.5 or SDXL pipeline from Hugging Face)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original", torch_dtype=torch.float16)

    # Loading the model weights into the pipeline
    pipe.unet.load_state_dict(model_weights)
    pipe.vae.load_state_dict(model_weights)

    # Set the seed for reproducibility
    if seed == 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    torch.manual_seed(seed)
    generator = torch.manual_seed(seed)

    # Prepare prompts and model inputs
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch16")
    text_input = tokenizer(positive_prompt, return_tensors="pt", padding="max_length", truncation=True)

    # Generate image step by step
    images = []
    for step in range(steps):
        # We can simulate the generation process step by step by progressively generating image
        with torch.no_grad():
            image = pipe(prompt=positive_prompt, width=width, height=height, num_inference_steps=1, generator=generator)["sample"][0]
            images.append(image)

        # Update the progress bar
        if progress:
            progress((step + 1) / steps)

    if progress:
        progress(1)  # Mark the progress as complete

    # Return the last generated image (or any other you might want)
    return images[-1]

# Gradio Interface
def create_gradio_app():
    with gr.Blocks() as demo:
        # Gradio Components for loading models and parameters
        checkpoint_path = gr.Textbox(label="Base Model Path (.safetensors)", placeholder="Enter path to base model (.safetensors)")
        lora_path = gr.Textbox(label="LoRA Model Path (.safetensors)", placeholder="Enter path to LoRA model (.safetensors)")
        lora_weight = gr.Slider(label="LoRA Weight", minimum=0, maximum=1, step=0.01, value=0.5)
        positive_prompt = gr.Textbox(label="Positive Prompt", placeholder="Describe what you want to see")
        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Describe what you want to avoid")
        width = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=512)
        height = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=512)
        seed = gr.Number(label="Seed (0 for random)", value=0)
        steps = gr.Slider(label="Steps", minimum=10, maximum=100, step=1, value=50)
        output_image = gr.Image(label="Generated Image")
        progress_bar = gr.Progress()
        generate_button = gr.Button("Generate Image")

        # Generate button event
        generate_button.click(
            fn=generate_image,
            inputs=[checkpoint_path, lora_path, lora_weight, positive_prompt, negative_prompt, width, height, seed, steps],
            outputs=output_image
        )

    demo.launch()

# Run the app
if __name__ == "__main__":
    create_gradio_app()
