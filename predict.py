import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import heapq

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.model import ImageCaptioningModel, AttentionImageCaptioningModel
from src.dataset import load_features, load_captions, load_tokenizer
from src.config import CONFIG


class BeamSearchNode:
    """
    Represents a node in the beam search tree
    """
    def __init__(self, sequence: List[int], log_prob: float, hidden_state=None):
        self.sequence = sequence  # List of token IDs
        self.log_prob = log_prob  # Cumulative log probability
        self.hidden_state = hidden_state  # LSTM hidden state (if available)
        self.is_finished = False
    
    def __lt__(self, other):
        # For max heap (higher probability = higher priority)
        return self.log_prob > other.log_prob


def load_trained_model(model_path: str, model_type: str = 'baseline') -> ImageCaptioningModel:
    """
    Load the trained model from the specified path
    """
    print(f"Loading trained model from {model_path}...")
    
    # Create model instance based on type
    if model_type == 'attention':
        print("Loading attention-based model...")
        model = AttentionImageCaptioningModel(
            vocab_size=8779,  # From the training logs
            max_length=CONFIG.MAX_LEN,
            embedding_dim=256,
            lstm_units=512,
            image_feature_dim=2048,
            attention_units=256,
            dropout_rate=0.3
        )
    else:
        print("Loading baseline model...")
        model = ImageCaptioningModel(
            vocab_size=8779,  # From the training logs
            max_length=CONFIG.MAX_LEN,
            embedding_dim=256,
            lstm_units=512,
            image_feature_dim=2048,
            dropout_rate=0.3
        )
    
    # Load the trained weights
    model.load_model(model_path)
    
    print("✓ Model loaded successfully")
    return model


def generate_caption_greedy(model: ImageCaptioningModel, image_features: np.ndarray, 
                          tokenizer: Tokenizer, max_length: int = 34, save_attention: bool = False, attention_dir: str = "evaluation_results/attention_maps", image_stem: str = None) -> str:
    """
    Generate caption for an image using greedy search
    """
    # Start with the start token
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    caption = [start_token]
    
    # Ensure image features are properly shaped
    image_features = image_features.reshape(1, -1)
    
    # Check if this is an attention model
    is_attention_model = hasattr(model, 'get_attention_model')
    has_attention_extractor = hasattr(model, 'get_attention_weights') and save_attention
    attention_steps = []
    
    for _ in range(max_length - 1):
        # Pad the current caption sequence
        sequence = pad_sequences([caption], maxlen=max_length, padding='post')
        
        # Optionally collect attention weights
        if has_attention_extractor:
            try:
                attn = model.get_attention_weights(image_features, sequence)
                attention_steps.append(attn)
            except Exception:
                attention_steps = []
                has_attention_extractor = False

        # Predict next word
        prediction = model.get_model().predict([image_features, sequence], verbose=0)
        
        if is_attention_model:
            # Attention model outputs sequences, get the last timestep
            predicted_word_id = np.argmax(prediction[0, len(caption)-1])
        else:
            # Baseline model outputs single word
            predicted_word_id = np.argmax(prediction[0])
        
        # Check if we hit the end token
        if predicted_word_id == end_token:
            break
            
        caption.append(predicted_word_id)
    
    # Convert to text
    caption_text = tokenizer.sequences_to_texts([caption])[0]
    
    # Clean up the caption
    caption_text = caption_text.replace('<start>', '').replace('<end>', '').strip()
    
    # Save attention if collected
    if save_attention and is_attention_model and image_stem is not None and len(attention_steps) > 0:
        try:
            Path(attention_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(attention_dir) / f"{image_stem}_attention.npy", np.array(attention_steps))
        except Exception:
            pass
    return caption_text


def generate_caption_beam(model: ImageCaptioningModel, image_features: np.ndarray, 
                        tokenizer: Tokenizer, beam_size: int = 3, max_length: int = 34,
                        save_attention: bool = False, attention_dir: str = "evaluation_results/attention_maps", image_stem: str = None) -> str:
    """
    Generate caption for an image using beam search
    """
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    
    # Ensure image features are properly shaped
    image_features = image_features.reshape(1, -1)
    
    # Initialize beam with start token
    initial_node = BeamSearchNode(sequence=[start_token], log_prob=0.0)
    beam = [initial_node]
    finished_sequences = []
    
    attention_steps = []
    for step in range(max_length - 1):
        candidates = []
        
        # Expand each sequence in the current beam
        for node in beam:
            if node.is_finished:
                continue
                
            # Pad the current sequence
            sequence = pad_sequences([node.sequence], maxlen=max_length, padding='post')
            
            # Predict next word probabilities
            prediction = model.get_model().predict([image_features, sequence], verbose=0)
            # Optionally collect attention weights from the current top node only
            if save_attention and hasattr(model, 'get_attention_weights') and node is beam[0]:
                try:
                    attn = model.get_attention_weights(image_features, sequence)
                    attention_steps.append(attn)
                except Exception:
                    attention_steps = []
            
            # Check if this is an attention model
            is_attention_model = hasattr(model, 'get_attention_model')
            if is_attention_model:
                # Attention model outputs sequences, get the last timestep
                word_probs = prediction[0, len(node.sequence)-1]  # Shape: (vocab_size,)
            else:
                # Baseline model outputs single word
                word_probs = prediction[0]  # Shape: (vocab_size,)
            
            # Get top-k candidates for this sequence
            top_k_indices = np.argsort(word_probs)[-beam_size:][::-1]
            
            for word_id in top_k_indices:
                word_prob = word_probs[word_id]
                log_prob = node.log_prob + np.log(word_prob + 1e-8)  # Add small epsilon to avoid log(0)
                
                new_sequence = node.sequence + [word_id]
                new_node = BeamSearchNode(sequence=new_sequence, log_prob=log_prob)
                
                # Check if sequence is finished
                if word_id == end_token:
                    new_node.is_finished = True
                    finished_sequences.append(new_node)
                else:
                    candidates.append(new_node)
        
        # Select top beam_size candidates
        if candidates:
            beam = heapq.nlargest(beam_size, candidates)
        else:
            break
    
    # Add any remaining unfinished sequences to finished sequences
    finished_sequences.extend(beam)
    
    # Return the sequence with highest probability
    if finished_sequences:
        best_sequence = max(finished_sequences, key=lambda x: x.log_prob)
    else:
        # Fallback to greedy if no sequences finished
        return generate_caption_greedy(model, image_features, tokenizer, max_length)
    
    # Convert to text
    caption_text = tokenizer.sequences_to_texts([best_sequence.sequence])[0]
    
    # Clean up the caption
    caption_text = caption_text.replace('<start>', '').replace('<end>', '').strip()
    
    # Save attention if collected
    if save_attention and image_stem is not None and len(attention_steps) > 0:
        try:
            Path(attention_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(attention_dir) / f"{image_stem}_attention.npy", np.array(attention_steps))
        except Exception:
            pass
    return caption_text


def predict_single_image(model: ImageCaptioningModel, image_path: str, tokenizer: Tokenizer, 
                        method: str = 'greedy', beam_size: int = 3,
                        save_attention: bool = False, attention_dir: str = "evaluation_results/attention_maps") -> str:
    """
    Predict caption for a single image
    """
    # Load and preprocess image
    from PIL import Image
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    
    # Load ResNet50 for feature extraction
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = resnet.predict(img_array, verbose=0)
    
    # Generate caption
    stem = Path(image_path).stem
    if method == 'beam':
        caption = generate_caption_beam(model, features, tokenizer, beam_size, save_attention=save_attention, attention_dir=attention_dir, image_stem=stem)
    else:
        caption = generate_caption_greedy(model, features, tokenizer, save_attention=save_attention, attention_dir=attention_dir, image_stem=stem)
    
    return caption


def predict_test_images(model: ImageCaptioningModel, test_features: Dict[str, np.ndarray], 
                       test_captions: Dict[str, List[str]], tokenizer: Tokenizer,
                       method: str = 'greedy', beam_size: int = 3,
                       save_attention: bool = False, attention_dir: str = "evaluation_results/attention_maps") -> Dict[str, str]:
    """
    Predict captions for test images using specified method
    """
    print(f"Generating captions using {method} search...")
    
    generated_captions = {}
    
    for i, (img_stem, image_features) in enumerate(test_features.items()):
        if i % 50 == 0:
            print(f"Processing image {i+1}/{len(test_features)}")
        
        # Find corresponding caption file name
        img_name = None
        for name in test_captions.keys():
            if Path(name).stem == img_stem:
                img_name = name
                break
        
        if img_name is None:
            continue
        
        # Generate caption
        stem = Path(img_name).stem
        if method == 'beam':
            generated_caption = generate_caption_beam(model, image_features, tokenizer, beam_size, save_attention=save_attention, attention_dir=attention_dir, image_stem=stem)
        else:
            generated_caption = generate_caption_greedy(model, image_features, tokenizer, save_attention=save_attention, attention_dir=attention_dir, image_stem=stem)
        
        generated_captions[img_name] = generated_caption
    
    return generated_captions


def compare_methods(model: ImageCaptioningModel, test_features: Dict[str, np.ndarray], 
                   test_captions: Dict[str, List[str]], tokenizer: Tokenizer,
                   beam_size: int = 3, max_images: int = 10) -> None:
    """
    Compare greedy vs beam search on a subset of images
    """
    print(f"\nComparing greedy vs beam search (beam_size={beam_size}) on {max_images} images...")
    
    # Limit to subset for comparison
    limited_features = dict(list(test_features.items())[:max_images])
    
    # Generate captions with both methods
    greedy_captions = predict_test_images(model, limited_features, test_captions, tokenizer, 'greedy')
    beam_captions = predict_test_images(model, limited_features, test_captions, tokenizer, 'beam', beam_size)
    
    # Show examples
    print("\n" + "="*80)
    print("COMPARISON: GREEDY vs BEAM SEARCH")
    print("="*80)
    
    for i, (img_name, greedy_caption) in enumerate(greedy_captions.items()):
        if i >= 5:  # Show only first 5 examples
            break
            
        beam_caption = beam_captions.get(img_name, "Not generated")
        reference_captions = test_captions.get(img_name, [])
        
        print(f"\nImage: {img_name}")
        print("-" * 50)
        print(f"Greedy:  {greedy_caption}")
        print(f"Beam:    {beam_caption}")
        print("Reference:")
        for j, ref in enumerate(reference_captions[:2], 1):  # Show first 2 references
            clean_ref = ref.replace('<start>', '').replace('<end>', '').strip()
            print(f"  {j}. {clean_ref}")


def main():
    """
    Main prediction pipeline
    """
    parser = argparse.ArgumentParser(description="Generate captions using CNN+LSTM model")
    parser.add_argument("--model_path", type=str, default="models/best_model.h5", 
                       help="Path to the trained model (default: models/best_model.h5)")
    parser.add_argument("--method", type=str, choices=['greedy', 'beam'], default='greedy',
                       help="Caption generation method (default: greedy)")
    parser.add_argument("--beam_size", type=int, default=3,
                       help="Beam size for beam search (default: 3)")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to a single image to caption")
    parser.add_argument("--test_subset", type=int, default=None,
                       help="Number of test images to process")
    parser.add_argument("--compare", action="store_true",
                       help="Compare greedy vs beam search methods")
    parser.add_argument("--model_type", type=str, choices=['baseline', 'attention'], default='baseline',
                       help="Model type: baseline or attention (default: baseline)")
    parser.add_argument("--save_attention", action="store_true",
                       help="Save attention maps during greedy/beam decoding (attention model only)")
    parser.add_argument("--attention_dir", type=str, default="evaluation_results/attention_maps",
                       help="Directory to save attention maps")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CNN+LSTM IMAGE CAPTIONING - CAPTION GENERATION")
    print("="*60)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(CONFIG.TOKENIZER_PATH)
    
    # Load trained model
    model = load_trained_model(args.model_path, args.model_type)
    
    if args.image_path:
        # Single image prediction
        if not Path(args.image_path).exists():
            print(f"Error: Image file not found at {args.image_path}")
            return
        
        print(f"Generating caption for: {args.image_path}")
        caption = predict_single_image(
            model, args.image_path, tokenizer, args.method, args.beam_size,
            save_attention=args.save_attention, attention_dir=args.attention_dir
        )
        
        print(f"\nGenerated Caption ({args.method}):")
        print(f"  {caption}")
        
    else:
        # Test dataset prediction
        print("Loading test data...")
        
        # Load test data
        with open("data/processed/splits/test_data.pkl", "rb") as f:
            test_data = pickle.load(f)
        
        # Filter features and captions for test set
        test_features = {}
        test_captions = {}
        
        for img_name, caps in test_data['captions'].items():
            img_stem = Path(img_name).stem
            if img_stem in test_data['features']:
                test_features[img_stem] = test_data['features'][img_stem]
                test_captions[img_name] = caps
        
        # Limit subset if specified
        if args.test_subset:
            test_features = dict(list(test_features.items())[:args.test_subset])
        
        if args.compare:
            # Compare both methods
            compare_methods(model, test_features, test_captions, tokenizer, args.beam_size)
        else:
            # Single method prediction
            generated_captions = predict_test_images(
                model, test_features, test_captions, tokenizer,
                args.method, args.beam_size,
                save_attention=args.save_attention, attention_dir=args.attention_dir
            )
            
            # Save results
            output_path = Path("prediction_results")
            output_path.mkdir(exist_ok=True)
            
            with open(output_path / f"captions_{args.method}.pkl", "wb") as f:
                pickle.dump(generated_captions, f)
            
            print(f"✓ Generated captions saved to {output_path / f'captions_{args.method}.pkl'}")
    
    print("\nCaption generation completed!")


if __name__ == "__main__":
    main()
