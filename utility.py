import numpy as np
from models_list import kaomojis
import os
import re


def modify_files_in_directory(directory, text, prepend=False):
    # Get a list of all files in the directory

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Iterate over each file
    for file_name in files:
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
    for file_name in files:
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
    for file_name in txt_files:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r') as file:
            content = file.read().strip().split(',')

        # Remove duplicates
        cleaned_list = [item.strip() for item in content]
        content = list(dict.fromkeys(cleaned_list))

        with open(file_path, 'w') as file:
            file.write(', '.join(content))

    return "Duplicates removed successfully"
