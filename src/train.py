import pickle
import glob
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import mixed_precision

from src.model import ImageCaptioningModel, AttentionImageCaptioningModel
from src.embeddings import get_embedding_matrix
from src.dataset_splitter import (
    load_split_files, split_dataset, create_sequences_for_split,
    save_split_data, load_split_data
)
from src.dataset import load_features, load_captions, load_tokenizer
from src.config import CONFIG


class EarlyStoppingReporter(Callback):
    """
    Custom callback to report when early stopping occurs and final metrics
    """
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_loss = logs.get('val_loss', float('inf'))
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            
    def on_train_end(self, logs=None):
        logs = logs or {}
        if self.stopped_epoch > 0:
            print("\n" + "="*60)
            print("EARLY STOPPING TRIGGERED!")
            print("="*60)
            print(f"✓ Training stopped at epoch {self.stopped_epoch + 1}")
            print(f"✓ Best validation loss: {self.best_val_loss:.6f} (epoch {self.best_epoch + 1})")
            train_acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            print(f"✓ Final training accuracy: {train_acc:.4f}" if isinstance(train_acc, (int, float)) else f"✓ Final training accuracy: {train_acc}")
            print(f"✓ Final validation accuracy: {val_acc:.4f}" if isinstance(val_acc, (int, float)) else f"✓ Final validation accuracy: {val_acc}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("TRAINING COMPLETED NORMALLY")
            print("="*60)
            train_acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            print(f"✓ Final training accuracy: {train_acc:.4f}" if isinstance(train_acc, (int, float)) else f"✓ Final training accuracy: {train_acc}")
            print(f"✓ Final validation accuracy: {val_acc:.4f}" if isinstance(val_acc, (int, float)) else f"✓ Final validation accuracy: {val_acc}")
            print("="*60)


def detect_and_print_device():
    """
    Detect and print which device TensorFlow is using
    """
    print("DEVICE DETECTION:")
    print("-" * 40)
    
    # Check available devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    
    print(f"Available devices:")
    print(f"  CPUs: {len(cpus)}")
    print(f"  GPUs: {len(gpus)}")
    
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Try to enable memory growth for GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Warning: Could not enable GPU memory growth: {e}")
    else:
        print("✗ No GPU found, using CPU")
    
    # Test which device will be used
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        device_used = test_tensor.device
        print(f"TensorFlow will use: {device_used}")
    
    print("-" * 40)
    return len(gpus) > 0


def prepare_training_data(project_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training, validation, and test data
    """
    print("Loading data...")
    
    # Load features, captions, and tokenizer
    features = load_features(CONFIG.FEATURES_PATH)
    captions = load_captions(CONFIG.CAPTIONS_MAP_PATH)
    tokenizer = load_tokenizer(CONFIG.TOKENIZER_PATH)
    
    # Load split files
    train_images, dev_images, test_images = load_split_files(project_root)
    
    # Split dataset
    train_data, dev_data, test_data = split_dataset(
        features, captions, train_images, dev_images, test_images
    )
    
    # Save split data for future use
    save_split_data(train_data, dev_data, test_data, Path("data/processed/splits"))
    
    print("Creating training sequences...")
    
    # Create sequences for each split
    X1_train, X2_train, y_train = create_sequences_for_split(
        tokenizer, train_data, CONFIG.MAX_LEN
    )
    
    X1_dev, X2_dev, y_dev = create_sequences_for_split(
        tokenizer, dev_data, CONFIG.MAX_LEN
    )
    
    X1_test, X2_test, y_test = create_sequences_for_split(
        tokenizer, test_data, CONFIG.MAX_LEN
    )
    
    print(f"Training samples: {len(X1_train)}")
    print(f"Validation samples: {len(X1_dev)}")
    print(f"Test samples: {len(X1_test)}")
    
    return X1_train, X2_train, y_train, X1_dev, X2_dev, y_dev, X1_test, X2_test, y_test, tokenizer


def setup_gpu_and_mixed_precision(use_mixed_precision: bool = True):
    """
    Setup GPU acceleration and mixed precision training
    """
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU acceleration enabled ({len(gpus)} GPU(s) detected)")
            
            # Enable mixed precision if requested
            if use_mixed_precision:
                try:
                    mixed_precision.set_global_policy("mixed_float16")
                    print("✓ Mixed precision training enabled (mixed_float16)")
                    return True, True  # (gpu_available, mixed_precision_enabled)
                except Exception as e:
                    print(f"✗ Could not enable mixed precision: {e}")
                    print("✓ Using float32 on GPU")
                    return True, False
            else:
                print("✗ Mixed precision disabled by user")
                return True, False
        except Exception as e:
            print(f"✗ GPU configuration error: {e}")
            print("✓ Falling back to CPU")
            return False, False
    else:
        print("✗ No GPU found, using CPU")
        if use_mixed_precision:
            print("✗ Mixed precision not available on CPU, using float32")
        return False, False


def create_callbacks(
    model_save_path: str,
    models_dir: str,
    patience: int = 5,
    monitor: str = 'val_loss',
    mode: str = 'min',
    epoch_prefix: str = "model_weights_epoch_"
) -> list:
    """
    Create training callbacks for model checkpointing and early stopping
    """
    # Create directories
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model - exactly as requested
        ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            verbose=1
        ),
        # Save model weights every epoch for resuming
        ModelCheckpoint(
            filepath=models_dir + f"/{epoch_prefix}{{epoch:02d}}.h5",
            monitor=monitor,
            save_best_only=False,
            save_weights_only=False,
            mode=mode,
            save_freq='epoch',
            verbose=1
        ),
        # Early stopping - exactly as requested
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        # Custom reporter for early stopping notifications
        EarlyStoppingReporter(),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            mode=mode,
            verbose=1
        )
    ]
    return callbacks


def find_latest_checkpoint(models_dir: str) -> Tuple[Optional[str], int]:
    """
    Find the latest checkpoint file in the models directory and return epoch number
    Returns: (checkpoint_path, epoch_number)
    """
    # Look for model weights files with different naming patterns
    patterns = [
        f"{models_dir}/model_weights_epoch_*.h5",
        f"{models_dir}/checkpoint_*.h5", 
        f"{models_dir}/model_weights.h5",
        f"{models_dir}/checkpoint.h5"
    ]
    
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(glob.glob(pattern))
    
    if not checkpoint_files:
        return None, 0
    
    # Sort by modification time (most recent first) and then by epoch number
    checkpoint_files.sort(key=lambda x: (Path(x).stat().st_mtime, int(x.split('_')[-1].split('.')[0]) if '_epoch_' in x else 0), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    # Extract epoch number from filename
    if '_epoch_' in latest_checkpoint:
        epoch_number = int(latest_checkpoint.split('_')[-1].split('.')[0])
    else:
        # For files without epoch numbers, try to determine from file modification time
        # This is a fallback - ideally all checkpoints should have epoch numbers
        epoch_number = 0
    
    return latest_checkpoint, epoch_number


def load_checkpoint_if_exists(model, models_dir: str, resume_enabled: bool) -> int:
    """
    Load checkpoint if it exists and return the epoch to resume from
    Returns: initial_epoch (0 if no checkpoint found)
    """
    if not resume_enabled:
        print("✗ Resume disabled by user")
        print("✓ Starting training from scratch")
        return 0
    
    latest_checkpoint, epoch_number = find_latest_checkpoint(models_dir)
    
    if latest_checkpoint:
        try:
            print(f"✓ Found checkpoint: {Path(latest_checkpoint).name}")
            print(f"✓ Loading model weights and optimizer state...")
            
            # Validate checkpoint file exists and is readable
            if not Path(latest_checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {latest_checkpoint}")
            
            if Path(latest_checkpoint).stat().st_size == 0:
                raise ValueError(f"Checkpoint file is empty: {latest_checkpoint}")
            
            # Load the complete model (weights + optimizer state)
            model.load_model(latest_checkpoint)
            
            print(f"✓ Successfully loaded checkpoint from epoch {epoch_number}")
            print(f"✓ Resuming training from epoch {epoch_number + 1}")
            return epoch_number + 1  # Resume from next epoch
            
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            print("✓ Attempting to find alternative checkpoint...")
            
            # Try to find other checkpoint files
            all_checkpoints = []
            for pattern in [f"{models_dir}/model_weights_epoch_*.h5", f"{models_dir}/checkpoint_*.h5"]:
                all_checkpoints.extend(glob.glob(pattern))
            
            if all_checkpoints:
                # Try the most recent alternative checkpoint
                all_checkpoints.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                for alt_checkpoint in all_checkpoints:
                    if alt_checkpoint != latest_checkpoint:
                        try:
                            print(f"✓ Trying alternative checkpoint: {Path(alt_checkpoint).name}")
                            model.load_model(alt_checkpoint)
                            alt_epoch = int(alt_checkpoint.split('_')[-1].split('.')[0]) if '_epoch_' in alt_checkpoint else 0
                            print(f"✓ Successfully loaded alternative checkpoint from epoch {alt_epoch}")
                            print(f"✓ Resuming training from epoch {alt_epoch + 1}")
                            return alt_epoch + 1
                        except Exception as alt_e:
                            print(f"✗ Alternative checkpoint also failed: {alt_e}")
                            continue
            
            print("✓ No valid checkpoints found, starting training from scratch")
            return 0
    else:
        print("✗ No checkpoint found in models/ directory")
        print("✓ Starting training from scratch")
        return 0


def cleanup_old_checkpoints(models_dir: str, keep_last_n: int = 5):
    """
    Clean up old checkpoint files to save disk space, keeping only the last N checkpoints
    """
    checkpoint_files = glob.glob(f"{models_dir}/model_weights_epoch_*.h5")
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Remove old checkpoints, keeping the last N
    files_to_remove = checkpoint_files[:-keep_last_n]
    for file_path in files_to_remove:
        try:
            Path(file_path).unlink()
            print(f"✓ Cleaned up old checkpoint: {Path(file_path).name}")
        except Exception as e:
            print(f"✗ Could not remove {Path(file_path).name}: {e}")


def display_checkpoint_status(models_dir: str, resume_enabled: bool):
    """
    Display checkpoint status at startup
    """
    print("\nCHECKPOINT STATUS:")
    print("-" * 50)
    
    if not resume_enabled:
        print("✗ Resume disabled by user")
        print("✓ Will start fresh training")
        print("-" * 50)
        return
    
    latest_checkpoint, epoch_number = find_latest_checkpoint(models_dir)
    
    if latest_checkpoint:
        print(f"✓ Found latest checkpoint: {Path(latest_checkpoint).name}")
        print(f"✓ Checkpoint contains: model weights + optimizer state")
        print(f"✓ Will resume from epoch: {epoch_number + 1}")
        print(f"✓ Checkpoint path: {latest_checkpoint}")
        
        # Show total number of checkpoints
        all_checkpoints = glob.glob(f"{models_dir}/model_weights_epoch_*.h5")
        print(f"✓ Total checkpoints available: {len(all_checkpoints)}")
    else:
        print("✗ No checkpoint found in models/ directory")
        print("✓ Will start fresh training from epoch 0")
        print("✓ Will save checkpoints every epoch")
    
    print("-" * 50)


def train_model(
    X1_train: np.ndarray,
    X2_train: np.ndarray,
    y_train: np.ndarray,
    X1_dev: np.ndarray,
    X2_dev: np.ndarray,
    y_dev: np.ndarray,
    tokenizer: Tokenizer,
    epochs: int = 20,
    batch_size: int = 32,
    resume_from_checkpoint: bool = True,
    use_mixed_precision: bool = True,
    model_type: str = 'baseline'
) -> ImageCaptioningModel:
    """
    Train the image captioning model with checkpointing and resuming capability
    """
    print("Building model...")
    
    # Setup GPU and mixed precision
    gpu_available, mixed_precision_enabled = setup_gpu_and_mixed_precision(use_mixed_precision)
    
    # Prepare embedding matrix if GloVe is enabled
    embedding_matrix = None
    if CONFIG.USE_GLOVE_EMBEDDINGS:
        try:
            print("Loading GloVe embeddings...")
            embedding_matrix = get_embedding_matrix(
                word_index=tokenizer.word_index,
                vocab_size=len(tokenizer.word_index) + 1,
                embedding_dim=CONFIG.GLOVE_EMBEDDING_DIM,
                glove_path=CONFIG.GLOVE_PATH,
                cache_path=Path("data/processed/embedding_matrix.pkl")
            )
            # Update embedding dimension to match GloVe
            embedding_dim = CONFIG.GLOVE_EMBEDDING_DIM
        except Exception as e:
            print(f"Warning: Failed to load GloVe embeddings: {e}")
            print("Falling back to random initialization...")
            embedding_matrix = None
            embedding_dim = 256
    else:
        print("Using random embeddings (GloVe disabled)")
        embedding_dim = 256
    
    # Create model based on type
    if model_type == 'attention':
        print("Creating attention-based model...")
        model = AttentionImageCaptioningModel(
            vocab_size=len(tokenizer.word_index) + 1,
            max_length=CONFIG.MAX_LEN,
            embedding_dim=embedding_dim,
            lstm_units=512,
            image_feature_dim=2048,
            attention_units=256,
            dropout_rate=0.3,
            embedding_matrix=embedding_matrix
        )
    else:
        print("Creating baseline model...")
        model = ImageCaptioningModel(
            vocab_size=len(tokenizer.word_index) + 1,
            max_length=CONFIG.MAX_LEN,
            embedding_dim=embedding_dim,
            lstm_units=512,
            image_feature_dim=2048,
            dropout_rate=0.3,
            embedding_matrix=embedding_matrix
        )
    
    # Print model summary
    model.summary()
    
    # Create directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load checkpoint if exists and set initial epoch
    initial_epoch = load_checkpoint_if_exists(model, str(models_dir), resume_from_checkpoint)
    
    # Create callbacks with appropriate model save path
    if model_type == 'attention':
        best_model_path = str(models_dir / "best_model_attention.h5")
        epoch_model_prefix = "model_attention_epoch_"
    else:
        best_model_path = str(models_dir / "best_model.h5")
        epoch_model_prefix = "model_weights_epoch_"
    
    callbacks = create_callbacks(
        model_save_path=best_model_path,
        models_dir=str(models_dir),
        patience=CONFIG.EARLY_STOPPING_PATIENCE,  # Early stopping patience from config
        monitor='val_loss',
        mode='min',
        epoch_prefix=epoch_model_prefix
    )
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training samples: {len(X1_train):,}")
    print(f"Validation samples: {len(X1_dev):,}")
    print(f"Batch size: {batch_size}")
    if initial_epoch > 0:
        print(f"Resuming from epoch: {initial_epoch}")
    else:
        print(f"Starting from epoch: {initial_epoch}")
    print(f"Total epochs: {epochs}")
    print(f"Device: {'GPU' if gpu_available else 'CPU'}")
    print(f"Mixed precision: {'Enabled' if mixed_precision_enabled else 'Disabled'}")
    print("="*60)
    print("Starting training...\n")
    
    # Train model
    history = model.get_model().fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_dev, X2_dev], y_dev),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch
    )
    
    # Save final model
    if model_type == 'attention':
        model.save_model(str(models_dir / "final_model_attention.h5"))
    else:
        model.save_model(str(models_dir / "final_model.h5"))
    
    # Save training history
    with open(models_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    
    # Clean up old checkpoints to save disk space
    print("\nCleaning up old checkpoints...")
    cleanup_old_checkpoints(str(models_dir), keep_last_n=5)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"✓ Best model saved to: {models_dir / 'best_model.h5'}")
    print(f"✓ Model weights saved to: {models_dir}")
    print(f"✓ Training history saved to: {models_dir / 'training_history.pkl'}")
    
    # Show final metrics from training history
    if hasattr(history, 'history') and history.history:
        final_train_acc = history.history.get('accuracy', [0])[-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        final_train_loss = history.history.get('loss', [0])[-1]
        final_val_loss = history.history.get('val_loss', [0])[-1]
        
        print(f"✓ Final Training Accuracy: {final_train_acc:.4f}")
        print(f"✓ Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"✓ Final Training Loss: {final_train_loss:.4f}")
        print(f"✓ Final Validation Loss: {final_val_loss:.4f}")
    
    print("="*60)
    return model


def evaluate_model(
    model: ImageCaptioningModel,
    X1_test: np.ndarray,
    X2_test: np.ndarray,
    y_test: np.ndarray
):
    """
    Evaluate the trained model on test data
    """
    print("Evaluating model on test data...")
    
    test_loss, test_accuracy = model.get_model().evaluate(
        [X1_test, X2_test], y_test,
        verbose=1
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


def main():
    """
    Main training pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Image Captioning Model")
    parser.add_argument("--epochs", type=int, default=CONFIG.MAX_EPOCHS, help=f"Number of training epochs (default: {CONFIG.MAX_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE, help=f"Batch size for training (default: {CONFIG.BATCH_SIZE})")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no_resume", action="store_true", help="Disable resuming from checkpoint")
    parser.add_argument("--model_type", type=str, choices=['baseline', 'attention'], default=CONFIG.MODEL_TYPE,
                       help=f"Model type: baseline or attention (default: {CONFIG.MODEL_TYPE})")
    parser.add_argument("--use_glove", action="store_true", help="Enable GloVe embeddings (overrides config)")
    parser.add_argument("--no_glove", action="store_true", help="Disable GloVe embeddings (overrides config)")
    
    args = parser.parse_args()
    
    # Override GloVe setting if specified via command line
    if args.use_glove:
        CONFIG.USE_GLOVE_EMBEDDINGS = True
    elif args.no_glove:
        CONFIG.USE_GLOVE_EMBEDDINGS = False
    
    project_root = Path(".")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Detect and print device information
    detect_and_print_device()
    
    print("="*60)
    print("CNN+LSTM IMAGE CAPTIONING MODEL TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  GloVe embeddings: {CONFIG.USE_GLOVE_EMBEDDINGS}")
    print(f"  Mixed precision: {not args.no_mixed_precision}")
    print(f"  Resume from checkpoint: {not args.no_resume}")
    print("="*60)
    
    # Display checkpoint status
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    display_checkpoint_status(str(models_dir), not args.no_resume)
    
    # Prepare data
    X1_train, X2_train, y_train, X1_dev, X2_dev, y_dev, X1_test, X2_test, y_test, tokenizer = prepare_training_data(project_root)
    
    # Train model
    model = train_model(
        X1_train, X2_train, y_train,
        X1_dev, X2_dev, y_dev,
        tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from_checkpoint=not args.no_resume,
        use_mixed_precision=not args.no_mixed_precision,
        model_type=args.model_type
    )
    
    # Evaluate model
    if len(X1_test) > 0:
        evaluate_model(model, X1_test, X2_test, y_test)
    
    print("Training pipeline completed!")


if __name__ == "__main__":
    main()
