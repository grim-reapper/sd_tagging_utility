import numpy as np
from models_list import kaomojis
import os
import re
from PIL import Image
from tqdm import tqdm
from caption_models import qwen_caption, florence_caption

def modify_files_in_directory(directory, text, prepend=False):
    # Get a list of all files in the directory

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Iterate over each file
    for file_name in tqdm(files):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)

            # Open the file for reading and writing
            with open(file_path, 'r+') as file:
                file_content = file.read()

                # Determine whether to prepend or append text
                if prepend:
                    file_content = text + " " + file_content
                else:
                    file_content += " " + text

                # Move cursor to the beginning of the file
                file.seek(0)

                # Write the modified content back to the file
                file.write(file_content)

    return "Task completed"


def search_and_replace(directory, search_pattern, replacement_text):
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Iterate over each file
    for file_name in tqdm(files):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)

            # Open the file for reading and writing
            with open(file_path, 'r+') as file:
                file_content = file.read()

                # Perform the search and replace operation
                updated_content = re.sub(search_pattern, replacement_text, file_content)

                # Move cursor to the beginning of the file
                file.seek(0)

                # Write the modified content back to the file
                file.write(updated_content)
                file.truncate()

    return "Task Completed"


def load_labels(dataframe):
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


def remove_duplicates(dir_path):
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    for file_name in tqdm(txt_files):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r') as file:
            content = file.read().strip().split(',')

        # Remove duplicates
        cleaned_list = [item.strip() for item in content]
        content = list(dict.fromkeys(cleaned_list))

        with open(file_path, 'w') as file:
            file.write(', '.join(content))

    return "Duplicates removed successfully"


def resize_images(input_dir, output_dir, width, height):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all files in the input directory
    files = os.listdir(input_dir)

    for file in tqdm(files):
        # Check if file is an image
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # Open image
            img = Image.open(os.path.join(input_dir, file))

            # Get current dimensions
            img_width, img_height = img.size

            # Calculate aspect ratio
            aspect_ratio = img_width / img_height

            # Check if image dimensions meet the requirement
            if img_width < width or img_height < height:
                # If image height is less than required height
                if img_height < height:
                    new_height = height
                    new_width = int(new_height * aspect_ratio)
                # If image width is less than required width
                elif img_width < width:
                    new_width = width
                    new_height = int(new_width / aspect_ratio)

                # Resize image while maintaining aspect ratio
                img = img.resize((new_width, new_height))

            else:
                # Resize image to fit within specified dimensions while maintaining aspect ratio
                if img_width / width > img_height / height:
                    new_width = width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = height
                    new_width = int(new_height * aspect_ratio)

                img = img.resize((new_width, new_height))

            # Save resized image
            img.save(os.path.join(output_dir, file))

    return "Image resized"


def create_caption(image, model):
    if image is not None:
        if model == "Florence-2":
            return florence_caption(image)
        elif model == "Qwen2-VL":
            return qwen_caption(image)
        # elif model == "JoyCaption":
        #     return joycaption(image)
    return ""