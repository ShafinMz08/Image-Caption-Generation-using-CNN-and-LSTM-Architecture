import os
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_resnet50_feature_extractor() -> tf.keras.Model:
    """
    Load pretrained ResNet50 with ImageNet weights, without the top classification layer.
    Uses global average pooling to return a 2048-d feature vector per image.
    """
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model


def _load_and_preprocess_image(img_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def extract_features(image_path: str, model: tf.keras.Model | None = None) -> np.ndarray:
    """
    Extract a single feature vector for the given image using ResNet50.

    If model is not provided, a new ResNet50 feature extractor is created.
    Returns a 1D numpy array of shape (2048,).
    """
    close_after = False
    if model is None:
        model = load_resnet50_feature_extractor()
        close_after = True
    batch = _load_and_preprocess_image(image_path)
    features = model.predict(batch, verbose=0).squeeze()
    if close_after:
        # Best effort cleanup
        try:
            del model
        except Exception:
            pass
    return features


def batch_extract_to_dir(
    images_dir: str,
    output_dir: str,
    batch_size: int = 16,
) -> None:
    """
    Batch process all .jpg images under images_dir, compute ResNet50 features,
    and save each as .npy in output_dir with the same stem name.
    """
    ensure_dir(output_dir)
    model = load_resnet50_feature_extractor()

    image_paths: List[str] = sorted(glob.glob(str(Path(images_dir) / "*.jpg")))
    if not image_paths:
        image_paths = sorted(str(p) for p in Path(images_dir).rglob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under: {images_dir}")

    batch_arrays: List[np.ndarray] = []
    batch_names: List[str] = []

    def flush_batch() -> None:
        if not batch_arrays:
            return
        batch = np.vstack(batch_arrays)
        feats = model.predict(batch, verbose=0)
        for name, vec in zip(batch_names, feats):
            out_path = Path(output_dir) / f"{name}.npy"
            np.save(out_path, vec.squeeze())

    for img_path in tqdm(image_paths, desc="Extracting ResNet50 features"):
        name = Path(img_path).stem
        out_file = Path(output_dir) / f"{name}.npy"
        if out_file.exists():
            continue

        arr = _load_and_preprocess_image(img_path)
        batch_arrays.append(arr)
        batch_names.append(name)

        if len(batch_arrays) == batch_size:
            flush_batch()
            batch_arrays.clear()
            batch_names.clear()

    # Flush remaining
    flush_batch()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ResNet50 feature extraction (ImageNet, no top layer)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .jpg images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/preprocessed_images",
        help="Directory to store extracted .npy feature files",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    batch_extract_to_dir(images_dir=args.input_dir, output_dir=args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()



