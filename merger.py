import torch
from safetensors.torch import load_file, save_file

# Paths to your files
sdxl_checkpoint_path = "F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL.safetensors"  # Existing SDXL checkpoint
vae_path = "F:/foocus/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors"  # VAE model weights (typically in PyTorch .pth format)
output_path = "F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL_vae.safetensors"  # Path to save the new checkpoint

# Load the SDXL checkpoint (in SafeTensors format)
print("Loading SDXL checkpoint...")
sdxl_checkpoint = load_file(sdxl_checkpoint_path)
# print(sdxl_checkpoint.keys())

# Load the VAE model weights
print("Loading SDXL VAE model...")
vae_state_dict = load_file(vae_path)

# Add the proper prefix for VAE (e.g., 'first_stage_model.')
vae_state_dict_with_prefix = {f"first_stage_model.{k}": v for k, v in vae_state_dict.items()}

# The SDXL checkpoint will have the model weights, we add the VAE to it
print("Baking VAE into the SDXL checkpoint...")
sdxl_checkpoint.update(vae_state_dict_with_prefix)  # Merge VAE state_dict into the SDXL checkpoint

# Save the new checkpoint with the VAE baked in
print("Saving the new SDXL checkpoint with VAE...")
save_file(sdxl_checkpoint, output_path)

print(f"SDXL checkpoint with VAE baked in saved to: {output_path}")


# Example usage
# sdxl_checkpoint_path = 'F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL.safetensors'
# vae_path = 'F:/foocus/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors'
# output_path = 'F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL_with_vae.safetensors'



# def merge_save_to_model():
#     # Load your Stable Diffusion model
#     model_path = r"F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL.safetensors"
#     pipe = StableDiffusionXLPipeline.from_pretrained(model_path)
#
#     # Load the SDXL VAE
#     vae_path = r"F:/foocus/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors"
#     vae = AutoencoderKL.from_pretrained(vae_path)
#
#     # Integrate the VAE into the pipeline
#     pipe.vae = vae
#
#     # Save the model
#     save_path = r"F:/foocus/stable-diffusion-webui/models/Stable-diffusion/IPDXL_VAE.safetensors"
#     torch.save(pipe.state_dict(), save_path)
#
# merge_save_to_model()