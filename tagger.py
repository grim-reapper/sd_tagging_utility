import gradio as gr
import glob
import os
from models_list import models, SWINV2_MODEL_DSV3_REPO
from predictor import Predictor
from utility import modify_files_in_directory, search_and_replace, remove_duplicates, resize_images, create_caption
import deepdanbooru as dd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from safetensors.torch import load_file, save_file

# os.environ['HF_HOME'] = 'F:\\huggingface'

def process_directory_for_caption(dir_path, modal, progress=gr.Progress()):
    # List to store the names of processed images
    processed_images = []

    # Check if the directory exists
    if not os.path.isdir(dir_path):
        return "The provided path is not a valid directory."

    # Get a list of all files in the directory
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_files = len(files)

    # Initialize the progress bar
    progress(0, desc="Processing images...")

    # Iterate over all image files in the directory
    for i, filename in enumerate(files):
        # Construct the full file path
        file_path = os.path.join(dir_path, filename)

        # Construct the corresponding .txt file path
        txt_file_path = os.path.splitext(file_path)[0] + '.txt'

        # Check if the .txt file already exists
        if not os.path.exists(txt_file_path):
            result = create_caption(file_path, modal)
            # Create the .txt file and write "hello world" to it
            with open(txt_file_path, 'w') as f:
                f.write(result)

            # Add the image name to the list of processed images
            processed_images.append(filename)

        # Update the progress bar
        progress((i + 1) / total_files, desc=f"Processed {i + 1}/{total_files} images")

    # Return a message indicating the number of images processed
    if processed_images:
        return f"Processed {len(processed_images)} images: {', '.join(processed_images)}"
    else:
        return "No new images found to process."

def extract_lora(checkpoint_path, output_name, progress=gr.Progress()):
    if not os.path.exists(checkpoint_path):
        return "Error: Checkpoint file not found."

    try:
        # Load the checkpoint
        progress(0, desc="Loading checkpoint...")
        checkpoint = load_file(checkpoint_path)

        # Function to identify LoRA weights based on naming conventions
        def is_lora_weight(key):
            # Common naming conventions for LoRA weights
            lora_keywords = ['lora', 'lora_up', 'lora_down', 'lora_alpha']
            return any(keyword in key for keyword in lora_keywords)

        # Extract LoRA weights
        progress(0.3, desc="Identifying LoRA weights...")
        lora_weights = {k: v for k, v in checkpoint.items() if is_lora_weight(k)}

        if not lora_weights:
            return "Error: No LoRA weights found in the checkpoint."

        # Save the extracted LoRA weights
        progress(0.7, desc="Saving LoRA weights...")
        output_path = f"{output_name}.safetensors"
        save_file(lora_weights, output_path)

        progress(1, desc="Extraction complete!")
        return f"LoRA weights extracted and saved to {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"


# Function to load LoRA models in safetensors format
def load_lora_model(lora_path):
    return load_file(lora_path)


# Function to resize tensors if shapes are different
def resize_tensor(tensor, target_shape):
    # Determine the smaller and larger shapes
    smaller_shape = min(tensor.shape, target_shape, key=lambda x: sum(x))
    larger_shape = max(tensor.shape, target_shape, key=lambda x: sum(x))

    # Resize the larger tensor to match the smaller one
    if tensor.shape == larger_shape:
        tensor = tensor[:smaller_shape[0], :smaller_shape[1]]

    return tensor


# Function to merge two LoRA models with progress callback
def merge_lora_models(*lora_models, alpha=0.5, progress=None):
    total_keys = sum(len(model.keys()) for model in lora_models)
    merged_lora = {}
    processed_keys = 0

    # Merging common keys
    all_keys = set().union(*(model.keys() for model in lora_models))
    for key in all_keys:
        tensors = [model.get(key) for model in lora_models if key in model]

        if tensors:
            # Ensure all tensors have the same shape or resize them
            target_shape = tensors[0].shape
            for i in range(1, len(tensors)):
                if tensors[i].shape != target_shape:
                    tensors[i] = resize_tensor(tensors[i], target_shape)

            # Merge the tensors
            merged_tensor = sum(alpha * tensor for tensor in tensors)
            merged_lora[key] = merged_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        processed_keys += 1
        if progress is not None:
            progress(processed_keys / total_keys)

    return merged_lora


# def merge_lora_models(lora_1, lora_2, alpha=0.5, progress=None):
#     total_keys = len(lora_1.keys()) + len(lora_2.keys())
#     merged_lora = {}
#     processed_keys = 0
#
#     # Merging common keys
#     for key in lora_1.keys():
#         tensor_1 = lora_1[key]
#
#         if key in lora_2:
#             tensor_2 = lora_2[key]
#             # Resize tensors if shapes are different
#             if tensor_1.shape != tensor_2.shape:
#                 tensor_1 = resize_tensor(tensor_1, tensor_2.shape)
#                 tensor_2 = resize_tensor(tensor_2, tensor_1.shape)
#
#             # Merge the tensors
#             merged_lora[key] = (alpha * tensor_1 + (1 - alpha) * tensor_2).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         else:
#             # If the key is not in lora_2, just keep tensor_1
#             merged_lora[key] = tensor_1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#         processed_keys += 1
#         if progress is not None:
#             progress(processed_keys / total_keys)
#
#     # Adding remaining keys from lora_2
#     for key in lora_2.keys():
#         if key not in merged_lora:
#             merged_lora[key] = lora_2[key].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         processed_keys += 1
#         if progress is not None:
#             progress(processed_keys / total_keys)
#
#     return merged_lora

# Function to save merged LoRA to safetensors format
def save_lora_model(merged_lora, output_path):
    save_file(merged_lora, output_path)


# Gradio interface function with progress bar
def merge_lora_files(lora_1_path, lora_2_path, lora_3_path=None, lora_4_path=None, output_path=None, alpha=0.5,
                     progress=gr.Progress(track_tqdm=True)):
    progress(0)  # Initialize progress
    try:
        # Load LoRA models
        lora_1 = load_lora_model(lora_1_path)
        lora_2 = load_lora_model(lora_2_path)
        lora_3 = load_lora_model(lora_3_path) if lora_3_path else None
        lora_4 = load_lora_model(lora_4_path) if lora_4_path else None

        # Filter out None values
        lora_models = [lora for lora in [lora_1, lora_2, lora_3, lora_4] if lora is not None]

        # Merge the LoRA models
        merged_lora = merge_lora_models(*lora_models, alpha=alpha, progress=progress)

        # Save the merged LoRA model
        save_lora_model(merged_lora, output_path)
        return f"Successfully merged and saved at: {output_path}"
    except Exception as e:
        return f"Error during merging: {str(e)}"


# def merge_lora_files(lora_1_path, lora_2_path, output_path, alpha, progress=gr.Progress(track_tqdm=True)):
#     progress(0)  # Initialize progress
#     try:
#         # Load LoRA models
#         lora_1 = load_lora_model(lora_1_path)
#         lora_2 = load_lora_model(lora_2_path)
#
#         # Merge the LoRA models
#         merged_lora = merge_lora_models(lora_1, lora_2, alpha, progress=progress)
#
#         # Save the merged LoRA model
#         save_lora_model(merged_lora, output_path)
#         return f"Successfully merged and saved at: {output_path}"
#     except Exception as e:
#         return f"Error during merging: {str(e)}"


# Function to load models
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model


def generate_caption(image, text=None, max_length=50, min_length=10, num_beams=4, no_repeat_ngram_size=2,
                     progress=gr.Progress()):
    if image is None:
        return
    processor, model = load_models()
    # Convert the image to RGB format
    raw_image = image.convert('RGB')

    progress(0, desc="Processing image...")

    if text:
        # Conditional image captioning
        inputs = processor(raw_image, text, return_tensors="pt")
    else:
        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

    progress(0.5, desc="Generating caption...")

    # Generate caption with specified parameters
    out = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size
    )
    caption = processor.decode(out[0], skip_special_tokens=True)

    progress(1, desc="Caption generated!")

    return caption


def process_directory(directory, text=None, max_length=50, min_length=10, num_beams=4, no_repeat_ngram_size=2,
                      progress=gr.Progress()):
    # List all image files in the directory
    image_files = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(
        os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.png'))

    total_images = len(image_files)
    progress(0, desc=f"Processing {total_images} images...")

    for i, image_path in enumerate(image_files):
        # Open the image
        image = Image.open(image_path)

        # Generate caption
        caption = generate_caption(image, text, max_length, min_length, num_beams, no_repeat_ngram_size, progress)

        # Save the caption to a .txt file with the same name as the image
        caption_file = os.path.splitext(image_path)[0] + '.txt'
        with open(caption_file, 'w') as f:
            f.write(caption)

        progress((i + 1) / total_images, desc=f"Processed {i + 1}/{total_images} images")

    progress(1, desc="All images processed!")


def load_model_weights(file_path):
    return load_file(file_path)


def save_model_weights(model_weights, output_path):
    save_file(model_weights, output_path)


def combine_sdxl_and_vae(sdxl_checkpoint_path, vae_path, output_path, progress=gr.Progress()):
    # Load the SDXL checkpoint
    sdxl_weights = load_model_weights(sdxl_checkpoint_path)
    progress(0.3, desc="Loaded SDXL checkpoint")

    # Load the VAE
    vae_state_dict = load_model_weights(vae_path)
    progress(0.6, desc="Loaded VAE")

    # Combine the weights
    vae_state_dict_with_prefix = {f"first_stage_model.{k}": v for k, v in vae_state_dict.items()}
    # The SDXL checkpoint will have the model weights, we add the VAE to it
    sdxl_weights.update(vae_state_dict_with_prefix)  # Merge VAE state_dict into the SDXL checkpoint
    progress(0.8, desc="Baking VAE into the SDXL checkpoint...")

    # Save the combined model
    save_file(sdxl_weights, output_path)
    progress(1.0, desc="Saved new SDXL checkpoint with VAE")

    return "Success: SDXL checkpoint with VAE baked in saved to " + output_path


def create_checkpoint_interface():
    with gr.Column():
        sdxl_checkpoint_input = gr.Textbox(label="SDXL Checkpoint Path",
                                           placeholder="path/to/sdxl_checkpoint.safetensors")
        vae_input = gr.Textbox(label="VAE Path", placeholder="path/to/sdxl_vae.safetensors")
        output_path_input = gr.Textbox(label="Output Path", placeholder="path/to/combined_sdxl_checkpoint.safetensors")

        combine_button = gr.Button("Combine Models")
        output_text = gr.Textbox(label="Result")

        combine_button.click(
            fn=combine_sdxl_and_vae,
            inputs=[sdxl_checkpoint_input, vae_input, output_path_input],
            outputs=output_text,
            api_name="combine_models"
        )

        progress_bar = gr.Progress(track_tqdm=True)


def load_model(model_path):
    return load_file(model_path)


# Function to pad tensors to match shapes
def pad_tensor(tensor, target_shape):
    pad_sizes = [(0, max(0, target_shape[i] - tensor.shape[i])) for i in range(len(tensor.shape))]
    padded_tensor = tensor
    for i in range(len(pad_sizes) - 1, -1, -1):
        pad = pad_sizes[i]
        padded_tensor = torch.nn.functional.pad(padded_tensor, (pad[0], pad[1]))
    return padded_tensor


# Function to match tensor shapes by padding or slicing
def match_tensor_shape(tensor, target_shape):
    slices = tuple(slice(0, min(tensor.shape[i], target_shape[i])) for i in range(len(tensor.shape)))
    matched_tensor = tensor[slices]
    if matched_tensor.shape != target_shape:
        matched_tensor = pad_tensor(matched_tensor, target_shape)
    return matched_tensor


# Function to merge two models with a given alpha
def merge_models(model_1, model_2, alpha=0.5, progress=None):
    total_keys = len(model_1.keys()) + len(model_2.keys())
    merged_model = {}
    processed_keys = 0

    for key in model_1.keys():
        tensor_1 = model_1[key]

        if key in model_2:
            tensor_2 = model_2[key]
            # Ensure tensors have the same shape
            if tensor_1.shape != tensor_2.shape:
                target_shape = tuple(max(tensor_1.shape[i], tensor_2.shape[i]) for i in range(len(tensor_1.shape)))
                tensor_1 = match_tensor_shape(tensor_1, target_shape)
                tensor_2 = match_tensor_shape(tensor_2, target_shape)
            # Merge tensors with alpha blending
            merged_model[key] = (alpha * tensor_1 + (1 - alpha) * tensor_2).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            merged_model[key] = tensor_1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        processed_keys += 1
        if progress is not None:
            progress(processed_keys / total_keys)

    for key in model_2.keys():
        if key not in merged_model:
            merged_model[key] = model_2[key].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        processed_keys += 1
        if progress is not None:
            progress(processed_keys / total_keys)

    return merged_model


# Function to save merged model in safetensors format
def save_model(merged_model, output_path):
    save_file(merged_model, output_path)


# Gradio interface function for merging models
def merge_safetensors_models(model_1_path, model_2_path, alpha, output_path, progress=gr.Progress(track_tqdm=True)):
    try:
        model_1 = load_model(model_1_path)
        model_2 = load_model(model_2_path)

        # Merge the models
        merged_model = merge_models(model_1, model_2, alpha, progress=progress)

        # Save the merged model
        save_model(merged_model, output_path)

        return f"Successfully merged and saved the model at: {output_path}"
    except Exception as e:
        return f"Error during merging: {str(e)}"


def main():
    predictor = Predictor()
    with gr.Blocks(title='Image Tagging utility') as iface:
        with gr.Tab("Interrogator"):
            with gr.Column():
                gr.Markdown(value=f"<h2>Generate prompt for image</h2>")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        image = gr.Image(type="pil", image_mode="RGBA", label="Input")
                        model_repo = gr.Dropdown(
                            models,
                            value=SWINV2_MODEL_DSV3_REPO,
                            label="Model",
                        )
                        with gr.Row():
                            general_thresh = gr.Slider(
                                0,
                                1,
                                step=0.05,
                                value=0.35,
                                label="General Tags Threshold",
                                scale=3,
                            )
                            general_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )

                        with gr.Row():
                            character_thresh = gr.Slider(
                                0,
                                1,
                                step=0.05,
                                value=0.85,
                                label="Character Tags Threshold",
                                scale=3,
                            )
                            character_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )

                        with gr.Row():
                            clear = gr.ClearButton(
                                components=[
                                    image,
                                    model_repo,
                                    general_thresh,
                                    general_mcut_enabled,
                                    character_thresh,
                                    character_mcut_enabled,
                                ],
                                variant='secondary',
                                size='lg',
                            )
                            submit = gr.Button(value='Submit', variant='primary', size='lg')

                    with gr.Column(variant='panel'):
                        sorted_general_strings = gr.Textbox(label='Output string')
                        rating = gr.Label(label='Rating')
                        character_res = gr.Label(label='Output (characters)')
                        general_res = gr.Label(label='Output (tags)')
                        clear.add(
                            [
                                sorted_general_strings,
                                rating,
                                character_res,
                                general_res,
                            ]
                        )

            submit.click(
                fn=predictor.predict,
                inputs=[
                    image,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                ],
                outputs=[sorted_general_strings, rating, character_res, general_res]
            )

        with gr.Tab("Tagging"):
            with gr.Column():
                with gr.Tab("WD Tagger"):
                    gr.Markdown(value=f"<h2>Tag images</h2>")
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            dir = gr.Textbox(label='Input Directory')
                            model_repo = gr.Dropdown(
                                models,
                                value=SWINV2_MODEL_DSV3_REPO,
                                label="Model",
                            )
                            with gr.Row():
                                general_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.35,
                                    label="General Tags Threshold",
                                    scale=3,
                                )
                                general_mcut_enabled = gr.Checkbox(
                                    value=False,
                                    label="Use MCut threshold",
                                    scale=1,
                                )

                            with gr.Row():
                                character_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.85,
                                    label="Character Tags Threshold",
                                    scale=3,
                                )
                                character_mcut_enabled = gr.Checkbox(
                                    value=False,
                                    label="Use MCut threshold",
                                    scale=1,
                                )
                            with gr.Row():
                                append_tags = gr.Checkbox(value=False, label="Append Tags",
                                                          info="Do you want to append tags to existing files?")

                            with gr.Row():
                                clear = gr.ClearButton(
                                    components=[
                                        dir,
                                        model_repo,
                                        general_thresh,
                                        general_mcut_enabled,
                                        character_thresh,
                                        character_mcut_enabled,
                                        append_tags
                                    ],
                                    variant='secondary',
                                    size='lg',
                                )
                                submit = gr.Button(value='Submit', variant='primary', size='lg')

                        with gr.Column(variant='panel'):
                            sorted_general_strings = gr.Textbox(label='Output string')
                            general_res = gr.HTML(label='Output (tags)')
                            clear.add(
                                [
                                    sorted_general_strings,
                                    general_res,
                                ]
                            )
                submit.click(
                    fn=predictor.label_images,
                    inputs=[
                        dir,
                        model_repo,
                        general_thresh,
                        general_mcut_enabled,
                        character_thresh,
                        character_mcut_enabled,
                        append_tags
                    ],
                    outputs=[sorted_general_strings, general_res]
                )
                with gr.Tab("DeepDanbooru Tagger"):
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            dir_name = gr.Textbox(label='Input Directory')
                            with gr.Row():
                                model_repo = gr.Dropdown(
                                    ['toynya/Z3D-E621-Convnext', 'model-resnet_custom_v3.h5'],
                                    value='toynya/Z3D-E621-Convnext',
                                    label="Model",
                                )
                            with gr.Row():
                                score_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.5,
                                    label="Score Threshold",
                                )
                            with gr.Row():
                                clear = gr.ClearButton(
                                    components=[
                                        dir_name,
                                        model_repo,
                                        score_thresh,
                                    ],
                                    variant='secondary',
                                    size='lg',
                                )
                                submit = gr.Button(value='Submit', variant='primary', size='lg')

                        with gr.Column(variant='panel'):
                            sorted_general_strings = gr.Textbox(label='Output string')
                            general_res = gr.HTML(label='Output (tags)')
                            clear.add(
                                [
                                    sorted_general_strings,
                                    general_res,
                                ]
                            )

                submit.click(
                    fn=predictor.predict_deepbooru,
                    inputs=[
                        dir_name,
                        model_repo,
                        score_thresh,
                    ],
                    outputs=[sorted_general_strings, general_res]
                )

                with gr.Tab("Salesforce Blip Captioning"):
                    with gr.Row():
                        image_input = gr.Image(type="pil", label="Upload an image")
                        text_input = gr.Textbox(label="Optional: Add a prefix for conditional captioning",
                                                placeholder="e.g., 'a photography of'")
                        directory_input = gr.Textbox(label="Directory containing images",
                                                     placeholder="e.g., /path/to/images")
                        max_length_input = gr.Slider(minimum=10, maximum=100, step=1, value=50,
                                                     label="Maximum Caption Length")
                        min_length_input = gr.Slider(minimum=10, maximum=100, step=1, value=10,
                                                     label="Minimum Caption Length")
                        num_beams_input = gr.Slider(minimum=1, maximum=10, step=1, value=4, label="Number of Beams")
                        no_repeat_ngram_size_input = gr.Slider(minimum=1, maximum=5, step=1, value=2,
                                                               label="No Repeat N-gram Size")
                        process_button = gr.Button("Process Directory")
                    caption_output = gr.Textbox(label="Generated Caption")

                    image_input.change(generate_caption,
                                       inputs=[image_input, text_input, max_length_input, min_length_input,
                                               num_beams_input, no_repeat_ngram_size_input], outputs=caption_output)
                    text_input.change(generate_caption,
                                      inputs=[image_input, text_input, max_length_input, min_length_input,
                                              num_beams_input, no_repeat_ngram_size_input], outputs=caption_output)
                    max_length_input.change(generate_caption,
                                            inputs=[image_input, text_input, max_length_input, min_length_input,
                                                    num_beams_input, no_repeat_ngram_size_input],
                                            outputs=caption_output)
                    min_length_input.change(generate_caption,
                                            inputs=[image_input, text_input, max_length_input, min_length_input,
                                                    num_beams_input, no_repeat_ngram_size_input],
                                            outputs=caption_output)
                    num_beams_input.change(generate_caption,
                                           inputs=[image_input, text_input, max_length_input, min_length_input,
                                                   num_beams_input, no_repeat_ngram_size_input], outputs=caption_output)
                    no_repeat_ngram_size_input.change(generate_caption,
                                                      inputs=[image_input, text_input, max_length_input,
                                                              min_length_input, num_beams_input,
                                                              no_repeat_ngram_size_input], outputs=caption_output)

                    process_button.click(process_directory,
                                         inputs=[directory_input, text_input, max_length_input, min_length_input,
                                                 num_beams_input, no_repeat_ngram_size_input], outputs=[])

                with gr.Tab("Generative AI Captioning"):
                    with gr.Row():
                        with gr.Column():
                          with gr.Row():
                            dir_input = gr.Textbox(label="Directory Path")
                          with gr.Row():
                            model = gr.Dropdown(["Florence-2", "Qwen2-VL", "JoyCaption"], label="Model", value="Florence-2")
                          with gr.Row():
                            process_button = gr.Button("Process Directory")
                        with gr.Column():
                            with gr.Row():
                                result_output = gr.Textbox(label="Result")

                # Define the button click event
                process_button.click(fn=process_directory_for_caption, inputs=[dir_input, model], outputs=result_output)

        with gr.Tab('Append/Prepend Tags'):
            with gr.Column():
                gr.Markdown(value=f"<h2>Generate prompt for image</h2>")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        dpath = gr.Textbox(label="Enter the path to the caption files directory")
                        ptext = gr.Textbox(label="Enter the text to append/prepend")
                        append = gr.Checkbox(label="Prepend tags?", value=True)
                    with gr.Column(variant='panel'):
                        output = gr.Textbox(label='Status')

            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=modify_files_in_directory, inputs=[dpath, ptext, append], outputs=output)

        with gr.Tab('Search/Replace Tag'):
            with gr.Row():
                dpath = gr.Textbox(label="Enter the path to the caption files directory")
                search = gr.Textbox(label="Text to search")
                replace = gr.Textbox(label="Text to replace with")
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=search_and_replace, inputs=[dpath, search, replace], outputs=output)

        with gr.Tab('Remove Duplicate Tags'):
            with gr.Row():
                dir_name = gr.Textbox(label='Input Directory')
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=remove_duplicates, inputs=[dir_name], outputs=output)

        with gr.Tab('Image Resizer'):
            with gr.Row():
                input_dir = gr.Textbox(label="Enter the path to the files directory")
            with gr.Row():
                output_dir = gr.Textbox(label="Enter the path to the files directory")
            with gr.Row():
                width = gr.Slider(100, 2000, step=50, label="Width", value=512)
                height = gr.Slider(100, 2000, step=50, label="Height", value=768)
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=resize_images, inputs=[input_dir, output_dir, width, height], outputs=output)

        with gr.Tab('Merge Loras'):
            with gr.Row():
                lora_1 = gr.File(label="LoRA Model 1 (.safetensors)")
            with gr.Row():
                lora_2 = gr.File(label="LoRA Model 2 (.safetensors)")
            with gr.Row():
                lora_3 = gr.File(label="LoRA Model 3 (.safetensors)")
            with gr.Row():
                lora_4 = gr.File(label="LoRA Model 4 (.safetensors)")
            with gr.Row():
                alpha = gr.Slider(0, 1, step=0.01, value=0.5, label="Merge Ratio (alpha)")
            with gr.Row():
                output_path = gr.Textbox(label="Output Path (with .safetensors extension)",
                                         placeholder="e.g., ./merged_model.safetensors")
            with gr.Row():
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Merge', variant='primary', size='lg')
            btn.click(fn=merge_lora_files, inputs=[lora_1, lora_2, lora_3, lora_4, output_path, alpha], outputs=output)

        with gr.Tab('Merge Checkpoints'):
            with gr.Row():
                model_1_dir = gr.Textbox(label="Path to Model 1", placeholder="Enter the path to Model 1 directory containing .safetensors")
            with gr.Row():
                model_2_dir = gr.Textbox(label="Path to Model 2", placeholder="Enter the path to Model 2 directory containing .safetensors")
            with gr.Row():
                output_dir = gr.Textbox(label="Output name", placeholder="Enter the directory where the merged model will be saved")
            with gr.Row():
                alpha = gr.Slider(0, 1, step=0.01, label="Merge Ratio (alpha)", value=0.5)
            with gr.Row():
                output = gr.Textbox(label="Output")
            with gr.Row():
                merge_button = gr.Button("Merge Models", variant='primary', size='lg')
                merge_button.click(fn=merge_safetensors_models, inputs=[model_1_dir, model_2_dir, alpha, output_dir], outputs=output)

        with gr.Tab('Lora Extractor'):
            with gr.Row():
                checkpoint_input = gr.Textbox(label="Safetensors checkpoint file path",
                                              placeholder="Enter Checkpoint File (safetensors) file")
            with gr.Row():
                output_name_input = gr.Textbox(label="Output Name",
                                               placeholder="Enter the name for the extracted LoRA file")
            with gr.Row():
                extract_button = gr.Button("Extract LoRA")
            with gr.Row():
                output_text = gr.Textbox(label="Output")

            extract_button.click(
                fn=extract_lora,
                inputs=[checkpoint_input, output_name_input],
                outputs=output_text
            )

        with gr.Tab('Bake in VAE into SDXL Model'):
            gr.Markdown("# SDXL Checkpoint and VAE Combiner")
            create_checkpoint_interface()

    iface.queue(max_size=1)
    iface.launch(debug=True)


if __name__ == '__main__':
    main()
