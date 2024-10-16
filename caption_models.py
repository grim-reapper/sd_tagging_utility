import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoModel, AutoTokenizer, \
    AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import numpy as np
import os
from datetime import datetime
import subprocess
import torch.nn as nn
os.environ['HF_HOME'] = 'F:\\huggingface'
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"},
               shell=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

#Initialize Florence model
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to(
    device).eval()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

# Initialize Qwen2-VL-2B model
# qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True,
#                                                              torch_dtype="auto").to(device).eval()
# qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

# Add these new imports and constants
CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
CHECKPOINT_PATH = "wpkklhc6"


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Load CLIP
# clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
# clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
# clip_model.eval()
# clip_model.requires_grad_(False)
# clip_model.to(device)
#
# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, token=HF_TOKEN)
#
# # LLM
# text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
#                                                   token=HF_TOKEN)
# text_model.eval()

# Image Adapter
# image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
# image_adapter.load_state_dict(torch.load(f"{CHECKPOINT_PATH}/image_adapter.pt", map_location="cpu"))
# image_adapter.eval()
# image_adapter.to(device)


def florence_caption(image):
    if isinstance(image, str):
        image = Image.open(image)  # Load the image from file path
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # Convert NumPy array to image
    # Convert grayscale image to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    return parsed_answer["<MORE_DETAILED_CAPTION>"]


def array_to_image_path(image_array):
    img = Image.fromarray(np.uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    full_path = os.path.abspath(filename)
    return full_path


def qwen_caption(image):
    pass
    # Check if 'image' is a file path string; if so, open the image
    # if isinstance(image, str):
    #     image = Image.open(image)  # Load the image from file path
    # elif not isinstance(image, Image.Image):
    #     image = Image.fromarray(np.uint8(image))  # Convert NumPy array to image
    #
    # # Convert the image to a file path
    # image_path = array_to_image_path(np.array(image))
    #
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": image_path,
    #             },
    #             {"type": "text", "text": "Describe this image in great detail in one paragraph."},
    #         ],
    #     }
    # ]
    #
    # text = qwen_processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = qwen_processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to(device)
    #
    # generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = qwen_processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    #
    # return output_text[0]

# def joycaption(image):
#     if not isinstance(image, Image.Image):
#         image = Image.fromarray(np.uint8(image))
#
#     # Preprocess image
#     image = clip_processor(images=image, return_tensors='pt').pixel_values
#     image = image.to(device)
#
#     # Tokenize the prompt
#     prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False,
#                               add_special_tokens=False)
#
#     # Embed image
#     with torch.amp.autocast_mode.autocast(device_type='cuda', enabled=True):
#         vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
#         image_features = vision_outputs.hidden_states[-2]
#         embedded_images = image_adapter(image_features)
#         embedded_images = embedded_images.to(device)
#
#     # Embed prompt
#     prompt_embeds = text_model.model.embed_tokens(prompt.to(device))
#     embedded_bos = text_model.model.embed_tokens(
#         torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=torch.int64))
#
#     # Construct prompts
#     inputs_embeds = torch.cat([
#         embedded_bos.expand(embedded_images.shape[0], -1, -1),
#         embedded_images.to(dtype=embedded_bos.dtype),
#         prompt_embeds.expand(embedded_images.shape[0], -1, -1),
#     ], dim=1)
#
#     input_ids = torch.cat([
#         torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
#         torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
#         prompt,
#     ], dim=1).to(device)
#     attention_mask = torch.ones_like(input_ids)
#
#     generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
#                                        max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5,
#                                        suppress_tokens=None)
#
#     # Trim off the prompt
#     generate_ids = generate_ids[:, input_ids.shape[1]:]
#     if generate_ids[0][-1] == tokenizer.eos_token_id:
#         generate_ids = generate_ids[:, :-1]
#
#     caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
#
#     return caption.strip()