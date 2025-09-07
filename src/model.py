import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, 
    Add, Concatenate, TimeDistributed, Lambda, 
    RepeatVector, Permute, Multiply, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from src.config import CONFIG
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class ImageCaptioningModel:
    """
    CNN+LSTM Image Captioning Model
    
    Architecture:
    1. Image features (pre-extracted ResNet50) -> Dense layer
    2. Caption sequences -> Embedding -> LSTM
    3. Concatenate image features with LSTM output
    4. Dense layers for vocabulary prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int = 256,
        lstm_units: int = 512,
        image_feature_dim: int = 2048,
        dropout_rate: float = 0.3,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.image_feature_dim = image_feature_dim
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the CNN+LSTM model architecture"""
        
        # Image feature input (pre-extracted ResNet50 features)
        image_input = Input(shape=(self.image_feature_dim,), name='image_input')
        image_dense = Dense(
            self.embedding_dim,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='image_dense'
        )(image_input)
        image_dropout = Dropout(self.dropout_rate)(image_dense)
        
        # Caption sequence input
        caption_input = Input(shape=(self.max_length,), name='caption_input')
        
        # Create embedding layer with or without pretrained weights
        if self.embedding_matrix is not None:
            print("Using pretrained GloVe embeddings")
            caption_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=True,  # Allow fine-tuning of embeddings
                mask_zero=True,
                name='caption_embedding'
            )(caption_input)
        else:
            print("Using randomly initialized embeddings")
            caption_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                mask_zero=True,
                name='caption_embedding'
            )(caption_input)
        
        caption_dropout = Dropout(self.dropout_rate)(caption_embedding)
        
        # LSTM layer - return only the last output (not sequences)
        lstm_layer = LSTM(
            self.lstm_units,
            return_sequences=False,
            name='lstm_layer'
        )(caption_dropout)
        lstm_dropout = Dropout(self.dropout_rate)(lstm_layer)
        
        # Concatenate image features with LSTM output
        combined = Concatenate(axis=-1)([image_dropout, lstm_dropout])
        
        # Dense layers for vocabulary prediction
        dense1 = Dense(
            self.lstm_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='dense1'
        )(combined)
        dense1_dropout = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(
            self.lstm_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='dense2'
        )(dense1_dropout)
        dense2_dropout = Dropout(self.dropout_rate)(dense2)
        
        # Output layer (vocabulary prediction) - single word output
        output = Dense(
            self.vocab_size,
            activation='softmax',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='output'
        )(dense2_dropout)
        
        # Create model
        self.model = Model(
            inputs=[image_input, caption_input],
            outputs=output,
            name='ImageCaptioningModel'
        )
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=CONFIG.LEARNING_RATE, clipnorm=5.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self) -> Model:
        """Return the compiled model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the model to filepath"""
        if self.model:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath, compile=False)
        print(f"Model loaded from: {filepath}")


class AttentionImageCaptioningModel:
    """
    CNN+LSTM Image Captioning Model with Bahdanau-style Attention
    
    Architecture:
    1. Image features (pre-extracted ResNet50) -> Dense layer
    2. Caption sequences -> Embedding -> LSTM with attention
    3. Attention mechanism over image features at each timestep
    4. Combine attended context with LSTM hidden state
    5. Dense layers for vocabulary prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int = 256,
        lstm_units: int = 512,
        image_feature_dim: int = 2048,
        attention_units: int = 256,
        dropout_rate: float = 0.3,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.image_feature_dim = image_feature_dim
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        
        self.model = None
        self.attention_model = None  # For extracting attention weights
        self._build_model()
    
    def _build_model(self):
        """Build the CNN+LSTM model with attention architecture"""
        
        # Image feature input (pre-extracted ResNet50 features)
        image_input = Input(shape=(self.image_feature_dim,), name='image_input')
        image_dense = Dense(
            self.embedding_dim,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='image_dense'
        )(image_input)
        image_dropout = Dropout(self.dropout_rate)(image_dense)
        
        # Caption sequence input
        caption_input = Input(shape=(self.max_length,), name='caption_input')
        
        # Create embedding layer with or without pretrained weights
        if self.embedding_matrix is not None:
            print("Using pretrained GloVe embeddings")
            caption_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=True,  # Allow fine-tuning of embeddings
                mask_zero=True,
                name='caption_embedding'
            )(caption_input)
        else:
            print("Using randomly initialized embeddings")
            caption_embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                mask_zero=True,
                name='caption_embedding'
            )(caption_input)
        
        caption_dropout = Dropout(self.dropout_rate)(caption_embedding)
        
        # LSTM layer - return only the last output (not sequences) for training compatibility
        lstm_layer = LSTM(
            self.lstm_units,
            return_sequences=False,
            name='lstm_layer'
        )(caption_dropout)
        lstm_dropout = Dropout(self.dropout_rate)(lstm_layer)
        
        # Attention mechanism - simplified for single word prediction
        # Create attention weights over image features
        attention_dense1 = Dense(self.attention_units, activation='tanh', name='attention_dense1')
        attention_dense2 = Dense(1, activation='linear', name='attention_dense2')
        
        # Calculate attention scores by combining LSTM output with image features
        # Repeat LSTM output to match image feature dimensions
        lstm_repeated = RepeatVector(1)(lstm_dropout)  # (batch, 1, lstm_units)
        image_repeated = RepeatVector(1)(image_dropout)  # (batch, 1, embedding_dim)
        
        # Combine LSTM and image features
        combined_features = Concatenate(axis=-1)([lstm_repeated, image_repeated])
        attention_scores = attention_dense1(combined_features)
        attention_scores = attention_dense2(attention_scores)
        
        # Apply softmax to get attention weights
        attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
        
        # Apply attention weights to image features
        # Weighted sum of image features
        context_vector = Multiply()([attention_weights, image_repeated])
        context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)  # Sum over timesteps
        
        # Combine LSTM output with attended context
        combined_output = Concatenate(axis=-1)([lstm_dropout, context_vector])
        
        # Dense layers for vocabulary prediction
        dense1 = Dense(
            self.lstm_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='dense1'
        )(combined_output)
        dense1_dropout = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(
            self.lstm_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='dense2'
        )(dense1_dropout)
        dense2_dropout = Dropout(self.dropout_rate)(dense2)
        
        # Output layer (vocabulary prediction) - single word output
        output = Dense(
            self.vocab_size,
            activation='softmax',
            kernel_regularizer=regularizers.l2(CONFIG.L2_REG),
            name='output'
        )(dense2_dropout)
        
        # Create main model
        self.model = Model(
            inputs=[image_input, caption_input],
            outputs=output,
            name='AttentionImageCaptioningModel'
        )
        
        # Create attention model for extracting attention weights
        self.attention_model = Model(
            inputs=[image_input, caption_input],
            outputs=attention_weights,
            name='AttentionWeightsModel'
        )
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=CONFIG.LEARNING_RATE, clipnorm=5.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self) -> Model:
        """Return the compiled model"""
        return self.model
    
    def get_attention_model(self) -> Model:
        """Return the attention weights model"""
        return self.attention_model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the model to filepath"""
        if self.model:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(filepath)
            print(f"Attention model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath, compile=False)
        print(f"Attention model loaded from: {filepath}")
    
    def get_attention_weights(self, image_features: np.ndarray, caption_sequence: np.ndarray) -> np.ndarray:
        """Extract attention weights for given inputs"""
        if self.attention_model is None:
            raise ValueError("Attention model not available")
        
        attention_weights = self.attention_model.predict([image_features, caption_sequence], verbose=0)
        return attention_weights


def create_callbacks(
    model_save_path: str,
    patience: int = 10,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> list:
    """
    Create training callbacks for model checkpointing and early stopping
    """
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=CONFIG.REDUCE_LR_FACTOR,
            patience=CONFIG.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            mode=mode,
            verbose=1
        )
    ]
    return callbacks


if __name__ == "__main__":
    # Example usage
    model = ImageCaptioningModel(
        vocab_size=10000,
        max_length=34,
        embedding_dim=256,
        lstm_units=512
    )
    model.summary()
