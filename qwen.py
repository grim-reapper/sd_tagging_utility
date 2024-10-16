import os
import gradio as gr
from caption_models import qwen_caption, florence_caption


# Set the HF_HOME environment variable
#os.environ['HF_HOME'] = 'C:\\path\\to\\your\\custom\\directory'

def create_caption(image, model):
    if image is not None:
        if model == "Florence-2":
            return florence_caption(image)
        elif model == "Qwen2-VL":
            return qwen_caption(image)
        # elif model == "JoyCaption":
        #     return joycaption(image)
    return ""
def process_directory(dir_path, progress=gr.Progress()):
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
            result = create_caption(file_path, "Florence-2")
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

# Create the Gradio interface
# Create the Gradio interface
with gr.Blocks() as iface:
    dir_input = gr.Textbox(label="Directory Path")
    process_button = gr.Button("Process Directory")
    result_output = gr.Textbox(label="Result")

    # Define the button click event
    process_button.click(fn=process_directory, inputs=dir_input, outputs=result_output)

# Launch the Gradio app
iface.launch()

#
# API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
# headers = {"Authorization": "Bearer YOUR_API_TOKEN"}
#
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.content
#
# prompt = "A little girl with pigtails and a pink dress is sitting on the floor, looking up at her mother with a sad and hungry expression. She is holding her tummy and saying, 'Mommy, I'm so hungry, can I please have some milk?' Her mother, standing nearby with a stern but concerned look, is holding a bottle of milk but hesitating to give it to her. The scene is set in a cozy living room with a warm, homey atmosphere."
#
# image_bytes = query({"inputs": prompt})
#
# # Save the generated image to a file
# with open("generated_image.png", "wb") as image_file:
#     image_file.write(image_bytes)
#
# print("Image generated and saved as generated_image.png")