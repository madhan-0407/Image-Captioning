import os
import re
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

# --- Constants and Configurations ---
image_folder = "/home/vglug/Documents/Image Captioning/Images"
caption_file = "/home/vglug/Documents/Image Captioning/captions.txt"
tokenizer_path = '/home/vglug/Documents/Image Captioning/tokenizer.pkl'
sequences_path = '/home/vglug/Documents/Image Captioning/sequences.pkl'
image_feature_path = "/home/vglug/Documents/Image Captioning/image_features.pkl"
max_caption_length = 36
img_height, img_width = 299, 299

# --- Function Definitions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = "<start> " + text + " <end>"
    return text

def extract_features(image_path, model):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    features = model.predict(img_array)
    return features.flatten()


def prepare_data():
    # -- Captions data --
    df = pd.read_csv(caption_file)
    df.columns = ["image", "caption"]
    df["caption"] = df["caption"].apply(clean_text)
    captions = {}
    for _, row in df.iterrows():
        image_id = row['image']
        caption = row['caption']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)

    # -- Images Data --
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    image_features = {}
    print("Extracting image features:")
    image_list = os.listdir(image_folder)
    for image_name in tqdm(image_list, desc="Processing Images"):
      image_path = os.path.join(image_folder, image_name)
      image_features[image_name] = extract_features(image_path, model)
    
    return captions, image_features

def tokenize_and_pad_captions(captions):
    # -- Tokenize data --
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([caption for captions_list in captions.values() for caption in captions_list])
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # -- Sequence Generation and Padding --
    sequences = {}
    for image_id, captions_list in captions.items():
      sequences[image_id] = tokenizer.texts_to_sequences(captions_list)
    
    return sequences, tokenizer

# --- Main Execution ---
print("Preparing data...")
captions, image_features = prepare_data()
print("Data preparation done.\n")

print("Tokenizing captions and generating padding sequences...")
sequences, tokenizer = tokenize_and_pad_captions(captions)
print("Tokenization done.\n")

# Optional: Save the image features
with open(image_feature_path, 'wb') as f:
   pickle.dump(image_features, f)
print(f"Image features saved in {image_feature_path}\n")

print("Saving the preprocessed data...")
# Save the sequences
with open('/home/vglug/Documents/Image Captioning/sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)
print("Preprocessed sequences saved in /content/sequences.pkl\n")

print("Data preparation step completed.")