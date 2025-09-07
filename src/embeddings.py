"""
GloVe embeddings utilities for the image captioning model.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle


def load_glove_embeddings(glove_path: Path, embedding_dim: int = 200) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a text file.
    
    Args:
        glove_path: Path to the GloVe embeddings file
        embedding_dim: Dimension of the embeddings
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    embeddings_index = {}
    
    if not glove_path.exists():
        print(f"Warning: GloVe file not found at {glove_path}")
        return embeddings_index
    
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index


def create_embedding_matrix(
    word_index: Dict[str, int], 
    embeddings_index: Dict[str, np.ndarray], 
    vocab_size: int, 
    embedding_dim: int
) -> np.ndarray:
    """
    Create embedding matrix for the vocabulary.
    
    Args:
        word_index: Tokenizer word index mapping
        embeddings_index: GloVe embeddings dictionary
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of embeddings
        
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    found_words = 0
    not_found_words = 0
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
            
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found_words += 1
        else:
            # Initialize with random values for words not in GloVe
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
            not_found_words += 1
    
    print(f"Found {found_words} words in GloVe embeddings.")
    print(f"Randomly initialized {not_found_words} words not found in GloVe.")
    
    return embedding_matrix


def save_embedding_matrix(embedding_matrix: np.ndarray, save_path: Path):
    """Save embedding matrix to file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(embedding_matrix, f)
    print(f"Embedding matrix saved to {save_path}")


def load_embedding_matrix(load_path: Path) -> np.ndarray:
    """Load embedding matrix from file."""
    if not load_path.exists():
        raise FileNotFoundError(f"Embedding matrix not found at {load_path}")
    
    with open(load_path, 'rb') as f:
        embedding_matrix = pickle.load(f)
    print(f"Embedding matrix loaded from {load_path}")
    return embedding_matrix


def get_embedding_matrix(
    word_index: Dict[str, int],
    vocab_size: int,
    embedding_dim: int,
    glove_path: Path,
    cache_path: Optional[Path] = None
) -> np.ndarray:
    """
    Get embedding matrix, loading from cache if available, otherwise creating from GloVe.
    
    Args:
        word_index: Tokenizer word index mapping
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of embeddings
        glove_path: Path to GloVe embeddings file
        cache_path: Optional path to cache the embedding matrix
        
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    # Try to load from cache first
    if cache_path and cache_path.exists():
        try:
            return load_embedding_matrix(cache_path)
        except Exception as e:
            print(f"Failed to load cached embedding matrix: {e}")
            print("Creating new embedding matrix from GloVe...")
    
    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
    
    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(
        word_index, embeddings_index, vocab_size, embedding_dim
    )
    
    # Save to cache if path provided
    if cache_path:
        save_embedding_matrix(embedding_matrix, cache_path)
    
    return embedding_matrix
