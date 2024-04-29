import os

import huggingface_hub
import numpy as np
import pandas as pd
import onnxruntime as ort
from models_list import LABEL_FILENAME, MODEL_FILENAME, kaomojis
from PIL import Image
import tensorflow as tf
import deepdanbooru as dd
# from utils import load_labels, mcut_threshold


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
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


def count_words(data_list):
    word_count = {}
    for word in data_list:
        word_count[word] = word_count.get(word, 0) + 1

    return word_count


class Predictor:
    def __init__(self):
        self.model = None
        self.model_target_size = None
        self.last_loaded_repo = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME, resume_download=True)
        model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME, resume_download=True)

        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = ort.InferenceSession(model_path)

        _, height, width, _ = model.get_inputs()[0].shape

        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
        # canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0])
        pad_top = (max_dim - image_shape[1])

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC
            )

        # convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def process_predictions(
            self,
            preds,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,

    ):
        labels = list(zip(self.tag_names, preds[0].astype(float)))
        # First 4 labels are actually ratings pick one
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)
        # then we have general tags pick any, where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any, where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        return sorted_general_strings, rating, character_res, general_res

    def predict(
            self,
            image,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
    ):
        self.load_model(model_repo)

        image = self.prepare_image(image)
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        return self.process_predictions(
            preds,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
        )

    def label_images(
            self,
            dir,
            model_repo,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
            append_tags
    ):
        self.load_model(model_repo)

        processed_files = []
        words_counting = []
        for image in os.listdir(dir):
            if image.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                img_path = os.path.join(dir, image)
                with Image.open(img_path) as r_image:
                    processed_image = self.prepare_image(r_image)
                    input_name = self.model.get_inputs()[0].name
                    # label_name = self.model.get_outputs()[0].name
                    preds = self.model.run(None, {input_name: processed_image})[0]
                    sorted_general_strings, rating, character_res, general_res = self.process_predictions(
                        preds,
                        general_thresh,
                        general_mcut_enabled,
                        character_thresh,
                        character_mcut_enabled,
                    )
                    words_counting.extend(list(general_res.keys()))
                    # rating_tags = [self.tag_names[i] for i in self.rating_indexes]
                    characters = ', '.join(character_res.keys())
                    final_tags_str = sorted_general_strings + ', ' + characters if characters else sorted_general_strings

                    caption_file_path = os.path.join(dir, f"{os.path.splitext(image)[0]}.txt")
                    if append_tags:
                        with open(caption_file_path) as file:
                            file_content = file.read()
                            final_tags_str = file_content + ", " + final_tags_str

                    with open(caption_file_path, 'w') as f:
                        f.write(final_tags_str)

                    processed_files.append(image)
        words = count_words(words_counting)
        # Create a list of formatted items
        words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
        formatted_items = [f"<li> {key} ({value})</li>" for key, value in words.items()]

        # Join the list items into a single string
        result = "".join(formatted_items)
        final_score = "<h1>Tags Occurrence</h1><ul style='list-style:none'>" + result + "</ul>"
        final_tags = "\n".join(processed_files)
        return final_tags, final_score

    def load_deepbooru_model(self):
        path = huggingface_hub.hf_hub_download(
            "public-data/DeepDanbooru",
            "model-resnet_custom_v3.h5",
            resume_download=True
        )
        model = tf.keras.models.load_model(path)
        return model

    def load_deepbooru_labels(self):
        path = huggingface_hub.hf_hub_download("public-data/DeepDanbooru", "tags.txt", resume_download=True)
        with open(path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def process_deepbooru_image(self, image, score_threshold):
        _, height, width, _ = self.booru_model.input_shape
        image = np.asarray(image)
        image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
        image = image.numpy()
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0
        probs = self.booru_model.predict(image[None, ...])[0]
        probs = probs.astype(float)

        indices = np.argsort(probs)[::-1]
        result_all = dict()
        result_threshold = dict()
        for index in indices:
            label = self.booru_labels[index]
            prob = probs[index]
            result_all[label] = prob
            if prob < score_threshold:
                break
            result_threshold[label] = prob
        result_text = ", ".join(result_all.keys())
        return result_threshold, result_all, result_text

    def predict_deepbooru(self, dir_name, score_threshold):
        self.booru_model = self.load_deepbooru_model()
        self.booru_labels = self.load_deepbooru_labels()

        processed_files = []
        words_counting = []

        for image in os.listdir(dir_name):
            if image.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                img_path = os.path.join(dir_name, image)
                with Image.open(img_path) as r_image:
                    result_threshold, result_all, result_text = self.process_deepbooru_image(r_image, score_threshold)

                    words_counting.extend(list(result_all.keys()))
                    caption_file_path = os.path.join(dir_name, f"{os.path.splitext(image)[0]}.txt")
                    with open(caption_file_path, 'w') as f:
                        f.write(result_text)

                    processed_files.append(image)
        words = count_words(words_counting)
        # Create a list of formatted items
        words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
        formatted_items = [f"<li> {key} ({value})</li>" for key, value in words.items()]

        # Join the list items into a single string
        result = "".join(formatted_items)
        final_score = "<h1>Tags Occurrence</h1><ul style='list-style:none'>" + result + "</ul>"
        final_tags = "\n".join(processed_files)
        return final_tags, final_score
