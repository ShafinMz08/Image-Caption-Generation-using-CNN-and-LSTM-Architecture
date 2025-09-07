import os
import re
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    # Import TensorFlow/Keras lazily so non-TF parts can run even if TF isn't installed yet
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
except Exception:  # pragma: no cover
    tf = None
    ResNet50 = None
    preprocess_input = None
    keras_image = None


# ------------------------------
# Filesystem utilities
# ------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_first_existing(paths: Iterable[str]) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    return ""


def auto_locate_captions_file(project_root: Path) -> str:
    candidates = [
        project_root / "data/raw/Flickr8k_text/Flickr8k.token.txt",
        project_root / "data/raw/Flickr8k/Flickr8k_text/Flickr8k.token.txt",
    ]
    # As a fallback, glob search
    globbed = list(project_root.glob("data/raw/**/Flickr8k.token.txt"))
    if globbed:
        candidates.extend(globbed)
    found = find_first_existing([str(p) for p in candidates])
    if not found:
        raise FileNotFoundError(
            "Could not locate Flickr8k.token.txt. Checked common locations under data/raw/."
        )
    return found


def auto_locate_images_dir(project_root: Path) -> str:
    candidates = [
        project_root / "data/raw/Flickr8k_Dataset",
        project_root / "data/raw/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset",
        project_root / "data/raw/Flickr8k/Flickr8k_Dataset/Flickr8k_Dataset",
    ]
    for c in candidates:
        if c.exists():
            # If it's a parent folder, ensure it actually contains images (directly or in subdirs)
            jpgs = list(c.rglob("*.jpg"))
            if jpgs:
                return str(c)
    # Fallback: find a directory under data/raw containing >1000 jpgs (Flickr8k has 8k)
    for p in project_root.glob("data/raw/**/"):
        if p.is_dir():
            jpgs = list(p.glob("*.jpg"))
            if len(jpgs) >= 1000:
                return str(p)
    raise FileNotFoundError(
        "Could not locate images directory. Expected under data/raw/; ensure JPGs are present."
    )


# ------------------------------
# Caption processing
# ------------------------------

_punct_regex = re.compile(r"[^a-z\s]")


def clean_caption(text: str) -> str:
    text = text.lower().strip()
    text = _punct_regex.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"<start> {text} <end>"


def parse_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """
    Parses Flickr8k.token.txt which has lines like:
        1000268201_693b08cb0e.jpg#0 A child in a pink dress ...

    Returns: dict mapping image filename (with extension) -> list of cleaned captions
    """
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                file_and_idx, raw_caption = line.split("\t", 1)
            except ValueError:
                # Some distributions use a single space; try a more permissive split
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    continue
                file_and_idx, raw_caption = parts
            img_filename = file_and_idx.split("#", 1)[0]
            cleaned = clean_caption(raw_caption)
            image_to_captions.setdefault(img_filename, []).append(cleaned)
    return image_to_captions


# ------------------------------
# Vocabulary
# ------------------------------

def build_vocabulary(image_to_captions: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build a vocabulary from all tokens present in the cleaned captions.
    Returns: (sorted_words_list, token_to_idx)
    """
    vocab_set = set()
    for captions in image_to_captions.values():
        for cap in captions:
            for token in cap.split():
                vocab_set.add(token)
    words = sorted(vocab_set)
    token_to_idx = {token: idx for idx, token in enumerate(words)}
    return words, token_to_idx


def save_pickle(obj, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ------------------------------
# Image feature extraction (ResNet50)
# ------------------------------

def load_resnet50_feature_extractor() -> "tf.keras.Model":
    if tf is None or ResNet50 is None:
        raise RuntimeError(
            "TensorFlow/Keras not available. Please install TensorFlow before running feature extraction."
        )
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return base_model


def load_and_preprocess_image(img_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def extract_features_for_images(images_dir: str, output_dir: str, batch_size: int = 1) -> None:
    ensure_dir(output_dir)
    model = load_resnet50_feature_extractor()

    # Collect image paths
    image_paths = sorted(glob.glob(str(Path(images_dir) / "*.jpg")))
    if not image_paths:
        # Try recursive
        image_paths = sorted(str(p) for p in Path(images_dir).rglob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under: {images_dir}")

    for img_path in tqdm(image_paths, desc="Extracting ResNet50 features"):
        img_name = Path(img_path).stem
        out_path = Path(output_dir) / f"{img_name}.npy"
        if out_path.exists():
            continue
        arr = load_and_preprocess_image(img_path)
        feats = model.predict(arr, verbose=0)
        np.save(out_path, feats.squeeze())


# ------------------------------
# Orchestration
# ------------------------------

def run_preprocessing(
    project_root: str = ".",
    captions_file: str = "",
    images_dir: str = "",
    processed_dir: str = "data/processed",
) -> None:
    root = Path(project_root).resolve()

    # Resolve inputs
    captions_file = captions_file or auto_locate_captions_file(root)
    images_dir = images_dir or auto_locate_images_dir(root)

    # Output paths
    vocab_path = str(root / processed_dir / "vocabulary.pkl")
    captions_map_path = str(root / processed_dir / "captions_map.pkl")
    image_features_dir = str(root / processed_dir / "preprocessed_images")

    print(f"Captions file: {captions_file}")
    print(f"Images dir:    {images_dir}")
    print(f"Outputs -> vocabulary: {vocab_path}")
    print(f"Outputs -> captions map: {captions_map_path}")
    print(f"Outputs -> image features dir: {image_features_dir}")

    # 1-3) Load and clean captions; map image -> captions
    image_to_captions = parse_flickr8k_captions(captions_file)

    # 4) Build vocabulary and save
    words, token_to_idx = build_vocabulary(image_to_captions)
    save_pickle({"words": words, "token_to_idx": token_to_idx}, vocab_path)

    # 6) Save image-to-caption mapping
    save_pickle(image_to_captions, captions_map_path)

    # 5) Extract and save image features
    extract_features_for_images(images_dir, image_features_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Flickr8k data for image captioning.")
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Project root directory (default: current working directory)",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="",
        help="Path to Flickr8k.token.txt (auto-detected if omitted)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="",
        help="Directory containing Flickr8k images (auto-detected if omitted)",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory to write outputs (vocabulary, captions map, features)",
    )

    args = parser.parse_args()
    run_preprocessing(
        project_root=args.project_root,
        captions_file=args.captions_file,
        images_dir=args.images_dir,
        processed_dir=args.processed_dir,
    )

import pandas as pd
import re
import string

def load_doc(filename):
    """
    Loads a document (text file) from the given filename.
    
    Args:
        filename (str): The path to the text file.
        
    Returns:
        str: The content of the file as a single string.
    """
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_descriptions(text):
    """
    Parses a text document containing image captions and organizes them
    into a dictionary.
    
    Args:
        text (str): The content of the captions text file.
        
    Returns:
        dict: A dictionary where keys are image IDs and values are
              a list of captions for that image.
    """
    mapping = {}
    # The Flickr8k captions file has each caption on a new line
    # with the format: 'image_id.jpg#caption_num caption_text'
    for line in text.strip().split('\n'):
        # Skip empty lines
        if len(line.split()) < 2:
            continue
        
        # Split the line into the image ID part and the caption text
        parts = line.split('.jpg#')
        image_id = parts[0]
        
        # The caption part is 'caption_num caption_text', so we split that
        caption_text = parts[1][2:]  # The '[2:]' removes the caption number
        
        # Add the caption to our dictionary
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption_text)
    return mapping

# --- Main script execution starts here ---
# Define the path to your captions file
# The path has been updated to match your file structure.
filename = 'data/raw/Flickr8k/Flickr8k_text' 

# Load the raw captions text
doc = load_doc(filename)

# Parse the text and store the descriptions in a dictionary
descriptions = load_descriptions(doc)

print(f"Loaded {len(descriptions)} unique images with captions.")
print("Example caption for an image:")
# Get the first image ID from the dictionary keys
first_image_id = list(descriptions.keys())[0]
print(f"Image ID: {first_image_id}")
print(f"Captions: {descriptions[first_image_id]}")

