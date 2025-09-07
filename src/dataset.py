from tensorflow.keras.preprocessing.text import Tokenizer
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import CONFIG


# ------------------------------
# Loaders (API as specified)
# ------------------------------

def load_features(features_path: Path) -> Dict[str, np.ndarray]:
    """
    Load ResNet50 features saved as individual .npy files in a directory OR
    a single pickle mapping {image_id: feature_vector}.
    Returns a dict {image_id: np.ndarray}.
    """
    path = Path(features_path)
    features: Dict[str, np.ndarray] = {}
    if path.is_dir():
        for file in path.glob("*.npy"):
            features[file.stem] = np.load(str(file))
        if not features:
            for file in path.rglob("*.npy"):
                features[file.stem] = np.load(str(file))
        if not features:
            raise FileNotFoundError(f"No .npy features found in directory: {path}")
        return features
    # File case: allow .pkl or .npy (single array is not supported here)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Pickle features must be a dict {image_id: feature_vector}")
        return obj
    raise ValueError("features_path must be a directory of .npy features or a .pkl mapping")


def load_tokenizer(tokenizer_path: Path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def load_captions(captions_map_path: Path) -> Dict[str, List[str]]:
    with open(captions_map_path, "rb") as f:
        captions = pickle.load(f)
    return captions


# ------------------------------
# Sequence generation
# ------------------------------

def create_sequences(
    tokenizer,
    max_len: int,
    captions_dict: Dict[str, List[str]],
    features_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training samples for next-word prediction from captions and image features.
    Returns arrays: X1 (image_features), X2 (padded input sequences), y (next word ids).
    """
    X1: List[np.ndarray] = []
    X2: List[List[int]] = []
    y: List[int] = []

    for img_id, caps in captions_dict.items():
        stem = Path(img_id).stem
        feat = features_dict.get(stem)
        if feat is None:
            continue
        seqs = tokenizer.texts_to_sequences(caps)
        for seq in seqs:
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_word = seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_len, padding="post")[0]
                X1.append(feat)
                X2.append(in_seq.tolist())
                y.append(out_word)

    if not X1:
        raise ValueError("No sequences generated. Check captions/features alignment and max_len.")

    X1_arr = np.stack(X1, axis=0)
    X2_arr = np.asarray(X2, dtype=np.int32)
    y_arr = np.asarray(y, dtype=np.int32)

    if CONFIG.VERBOSE:
        print(f"X1 (image features): {X1_arr.shape}")
        print(f"X2 (padded seqs):    {X2_arr.shape}")
        print(f"y  (targets):        {y_arr.shape}")

    return X1_arr, X2_arr, y_arr


# ------------------------------
# Orchestrator
# ------------------------------

def get_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features, tokenizer, and captions using CONFIG; build training arrays.
    Returns (X1, X2, y).
    """
    features = load_features(CONFIG.FEATURES_PATH)
    tokenizer = load_tokenizer(CONFIG.TOKENIZER_PATH)
    captions = load_captions(CONFIG.CAPTIONS_MAP_PATH)
    return create_sequences(tokenizer, CONFIG.MAX_LEN, captions, features)


if __name__ == "__main__":
    # Simple manual run/verification
    X1, X2, y = get_training_data()
    if CONFIG.VERBOSE:
        print("Training data prepared.")


