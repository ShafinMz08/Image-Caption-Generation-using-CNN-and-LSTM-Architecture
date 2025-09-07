import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import CONFIG


def load_split_files(project_root: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train, dev, and test image lists from the Flickr8k dataset split files.
    Returns: (train_images, dev_images, test_images)
    """
    data_dir = project_root / "data" / "raw" / "Flickr8k" / "Flickr8k_text"
    
    # Load image lists
    with open(data_dir / "Flickr_8k.trainImages.txt", "r") as f:
        train_images = [line.strip() for line in f.readlines()]
    
    with open(data_dir / "Flickr_8k.devImages.txt", "r") as f:
        dev_images = [line.strip() for line in f.readlines()]
    
    with open(data_dir / "Flickr_8k.testImages.txt", "r") as f:
        test_images = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(train_images)} training images")
    print(f"Loaded {len(dev_images)} dev images")
    print(f"Loaded {len(test_images)} test images")
    
    return train_images, dev_images, test_images


def split_dataset(
    features: Dict[str, np.ndarray],
    captions: Dict[str, List[str]],
    train_images: List[str],
    dev_images: List[str],
    test_images: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split the dataset into train, dev, and test sets based on image lists.
    Returns: (train_data, dev_data, test_data)
    """
    def create_split_data(image_list: List[str]) -> Dict[str, Any]:
        split_data = {
            'features': {},
            'captions': {}
        }
        
        for img_name in image_list:
            img_stem = Path(img_name).stem
            
            # Add features if available
            if img_stem in features:
                split_data['features'][img_stem] = features[img_stem]
            
            # Add captions if available
            if img_name in captions:
                split_data['captions'][img_name] = captions[img_name]
        
        return split_data
    
    train_data = create_split_data(train_images)
    dev_data = create_split_data(dev_images)
    test_data = create_split_data(test_images)
    
    print(f"Train data: {len(train_data['features'])} features, {len(train_data['captions'])} captions")
    print(f"Dev data: {len(dev_data['features'])} features, {len(dev_data['captions'])} captions")
    print(f"Test data: {len(test_data['features'])} features, {len(test_data['captions'])} captions")
    
    return train_data, dev_data, test_data


def create_sequences_for_split(
    tokenizer,
    split_data: Dict[str, Any],
    max_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training sequences for a specific split (train/dev/test).
    Returns: (X1, X2, y) where X1=image_features, X2=padded_sequences, y=target_words
    """
    X1: List[np.ndarray] = []
    X2: List[List[int]] = []
    y: List[int] = []
    
    features = split_data['features']
    captions = split_data['captions']
    
    for img_name, caps in captions.items():
        img_stem = Path(img_name).stem
        
        # Get image features
        if img_stem not in features:
            continue
            
        feat = features[img_stem]
        
        # Convert captions to sequences
        seqs = tokenizer.texts_to_sequences(caps)
        
        # Create training samples for each caption
        for seq in seqs:
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_word = seq[i]
                
                # Pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_len, padding="post")[0]
                
                X1.append(feat)
                X2.append(in_seq.tolist())
                y.append(out_word)
    
    if not X1:
        print(f"Warning: No sequences generated for this split")
        return np.array([]), np.array([]), np.array([])
    
    X1_arr = np.stack(X1, axis=0)
    X2_arr = np.asarray(X2, dtype=np.int32)
    y_arr = np.asarray(y, dtype=np.int32)
    
    return X1_arr, X2_arr, y_arr


def save_split_data(
    train_data: Dict[str, Any],
    dev_data: Dict[str, Any],
    test_data: Dict[str, Any],
    output_dir: Path
):
    """
    Save split data to pickle files for future use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(output_dir / "dev_data.pkl", "wb") as f:
        pickle.dump(dev_data, f)
    
    with open(output_dir / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"Split data saved to {output_dir}")


def load_split_data(splits_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load previously saved split data from pickle files.
    Returns: (train_data, dev_data, test_data)
    """
    with open(splits_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    with open(splits_dir / "dev_data.pkl", "rb") as f:
        dev_data = pickle.load(f)
    
    with open(splits_dir / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    print(f"Split data loaded from {splits_dir}")
    return train_data, dev_data, test_data
