# Tagging Utility for preparing dataset to train Stable Diffusion Models

This tagging utility is designed for stable diffusion models. It provides a user-friendly interface using the Gradio library and offers the following features:

1. Interrogate Image:
   - Generates tags for a single image using the WD Tagger.

2. Generate Tags for All Images in a Directory using WD Tagger:
   - Generates tags for all images available in a specified directory using the WD Tagger.

3. Generate Tags for All Images in a Directory using Deepbooru Tagger:
   - Generates tags for all images available in a specified directory using the Deepbooru Tagger.

4. Remove Duplicate Tags:
   - Removes duplicate tags from the generated tags.

5. Search and Replace Specific Tags from Files:
   - Allows searching for specific tags and replacing them in files.

6. Append/Prepend Tags to Existing Tags:
   - Appends or prepends tags to existing tags in files.

## How to Use:

1. **Installation:**

   - Clone the repository:
     ```
     git clone <git@github.com:grim-reapper/sd_tagging_utility.git>
     cd sd_tagging_utility
     ```

   - Install the required dependencies:
     ```
     pip install -r requirements.txt
     ```

2. **Running the Utility:**

   - Run the tagging utility script:
     ```
     python tagger.py
     ```

3. **Using the Interface:**

   - Once the utility is running, you will be presented with a user-friendly interface powered by Gradio.

   - Select the desired feature from the options available.

   - Follow the instructions provided in the interface to input the necessary information.

   - After processing, the utility will display the results and save them to the specified directory or file.

## Requirements:

- Python 3.10
- Gradio
- Other dependencies listed in `requirements.txt`

## Contributors:

- [Imran Ali]