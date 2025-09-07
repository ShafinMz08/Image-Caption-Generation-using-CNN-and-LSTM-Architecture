from pathlib import Path


class CONFIG:
    """
    Central configuration for dataset preparation and preprocessing.

    Adjust these paths to match your workspace layout. All consumers should import
    CONFIG rather than hardcoding paths.
    """

    # Directory containing ResNet50 .npy feature files OR a single pickle with a dict
    # mapping image_id -> feature vector. If a directory, features are expected as
    # one .npy per image with the image stem as the filename.
    FEATURES_PATH: Path = Path("data/processed/preprocessed_images")

    # Pickle containing {image_filename (or id): [list of tokenized captions with start/end tokens]}
    CAPTIONS_MAP_PATH: Path = Path("data/processed/captions_map.pkl")

    # Pickle containing a fitted Keras Tokenizer
    TOKENIZER_PATH: Path = Path("data/processed/tokenizer.pkl")

    # Maximum length for padded input sequences
    MAX_LEN: int = 34

    # If True, dataset preparation will print helpful shape information
    VERBOSE: bool = True
    
    # Model configuration
    MODEL_TYPE: str = "attention"  # "baseline" or "attention"
    
    # GloVe embeddings configuration
    USE_GLOVE_EMBEDDINGS: bool = True
    GLOVE_PATH: Path = Path("data/glove/glove.6B.200d.txt")
    GLOVE_EMBEDDING_DIM: int = 200
    
    # Training configuration
    MAX_EPOCHS: int = 80
    EARLY_STOPPING_PATIENCE: int = 7
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    REDUCE_LR_FACTOR: float = 0.5
    REDUCE_LR_PATIENCE: int = 3
    DROPOUT_RATE: float = 0.5
    L2_REG: float = 1e-4



