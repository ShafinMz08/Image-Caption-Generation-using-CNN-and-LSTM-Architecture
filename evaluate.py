import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU and METEOR scores will be calculated using alternative methods.")

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocoevalcap not available. CIDEr scores will not be calculated.")

from src.model import ImageCaptioningModel, AttentionImageCaptioningModel
from src.dataset import load_features, load_captions, load_tokenizer
from src.config import CONFIG

# Import beam search from predict.py
from predict import generate_caption_beam, BeamSearchNode


def load_test_data() -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], Tokenizer]:
    """
    Load test image features, captions, and tokenizer
    """
    print("Loading test data...")
    
    # Load features, captions, and tokenizer
    features = load_features(CONFIG.FEATURES_PATH)
    captions = load_captions(CONFIG.CAPTIONS_MAP_PATH)
    tokenizer = load_tokenizer(CONFIG.TOKENIZER_PATH)
    
    # Load test split
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
    
    print(f"Loaded {len(test_features)} test images with {len(test_captions)} caption sets")
    return test_features, test_captions, tokenizer


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
    
    print("Model loaded successfully")
    return model


def generate_caption_greedy(model: ImageCaptioningModel, image_features: np.ndarray, 
                          tokenizer: Tokenizer, max_length: int = 34) -> str:
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
    
    for _ in range(max_length - 1):
        # Pad the current caption sequence
        sequence = pad_sequences([caption], maxlen=max_length, padding='post')
        
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
    
    return caption_text


def generate_caption_greedy_with_attention(
    model: ImageCaptioningModel,
    image_features: np.ndarray,
    tokenizer: Tokenizer,
    max_length: int = 34,
):
    """
    Generate caption and collect attention weights per decoding step (if available).
    Returns (caption_text, attention_array or None)
    """
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    caption = [start_token]
    image_features = image_features.reshape(1, -1)

    attention_steps = []
    has_attention = hasattr(model, 'get_attention_weights')

    for _ in range(max_length - 1):
        sequence = pad_sequences([caption], maxlen=max_length, padding='post')

        if has_attention:
            try:
                attn = model.get_attention_weights(image_features, sequence)
                attention_steps.append(attn)
            except Exception:
                attention_steps = []
                has_attention = False

        prediction = model.get_model().predict([image_features, sequence], verbose=0)
        if hasattr(model, 'get_attention_model'):
            predicted_word_id = np.argmax(prediction[0, len(caption)-1])
        else:
            predicted_word_id = np.argmax(prediction[0])

        if predicted_word_id == end_token:
            break
        caption.append(predicted_word_id)

    caption_text = tokenizer.sequences_to_texts([caption])[0]
    caption_text = caption_text.replace('<start>', '').replace('<end>', '').strip()

    if has_attention and len(attention_steps) > 0:
        try:
            attn_arr = np.array(attention_steps)
        except Exception:
            attn_arr = None
        return caption_text, attn_arr
    return caption_text, None


def calculate_bleu_scores(reference: List[str], candidate: str) -> Dict[str, float]:
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if not NLTK_AVAILABLE:
        # Simple BLEU calculation without NLTK
        return calculate_simple_bleu(reference, candidate)
    
    # Tokenize reference and candidate
    ref_tokens = [ref.split() for ref in reference]
    cand_tokens = candidate.split()
    
    # Calculate BLEU scores with smoothing
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {}
    for n in range(1, 5):
        weights = [1/n] * n + [0] * (4-n)
        score = sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=smoothing)
        bleu_scores[f'BLEU-{n}'] = score
    
    return bleu_scores


def calculate_simple_bleu(reference: List[str], candidate: str) -> Dict[str, float]:
    """
    Simple BLEU calculation without NLTK
    """
    def get_ngrams(text: str, n: int) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def precision_score(candidate_ngrams: List[str], reference_ngrams_list: List[List[str]]) -> float:
        if not candidate_ngrams:
            return 0.0
        
        matches = 0
        for ngram in candidate_ngrams:
            for ref_ngrams in reference_ngrams_list:
                if ngram in ref_ngrams:
                    matches += 1
                    break
        
        return matches / len(candidate_ngrams)
    
    def brevity_penalty(candidate: str, references: List[str]) -> float:
        cand_len = len(candidate.split())
        ref_lens = [len(ref.split()) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - cand_len))
        
        if cand_len > closest_ref_len:
            return 1.0
        else:
            return np.exp(1 - closest_ref_len / cand_len)
    
    candidate_words = candidate.split()
    reference_words_list = [ref.split() for ref in reference]
    
    bleu_scores = {}
    for n in range(1, 5):
        candidate_ngrams = get_ngrams(candidate, n)
        reference_ngrams_list = [get_ngrams(ref, n) for ref in reference]
        
        prec = precision_score(candidate_ngrams, reference_ngrams_list)
        bp = brevity_penalty(candidate, reference)
        
        bleu_scores[f'BLEU-{n}'] = bp * prec
    
    return bleu_scores


def calculate_meteor_score(reference: List[str], candidate: str) -> float:
    """
    Calculate METEOR score
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        # Tokenize
        ref_tokens = [ref.split() for ref in reference]
        cand_tokens = candidate.split()
        
        # Calculate METEOR score
        score = meteor_score(ref_tokens, cand_tokens)
        return score
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_cider_score(references: Dict[str, List[str]], candidates: Dict[str, str]) -> float:
    """
    Calculate CIDEr score
    """
    if not COCO_EVAL_AVAILABLE:
        return 0.0
    
    try:
        cider = Cider()
        score, _ = cider.compute_score(references, candidates)
        return score
    except Exception as e:
        print(f"Error calculating CIDEr score: {e}")
        return 0.0


def evaluate_model(model: ImageCaptioningModel, test_features: Dict[str, np.ndarray], 
                  test_captions: Dict[str, List[str]], tokenizer: Tokenizer, 
                  method: str = 'greedy', beam_size: int = 3,
                  save_attention: bool = False, attention_dir: str = "evaluation_results/attention_maps") -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Evaluate the model on test data and calculate various metrics
    """
    print("Generating captions for test images...")
    
    generated_captions = {}
    if save_attention:
        Path(attention_dir).mkdir(parents=True, exist_ok=True)
    all_bleu_scores = {f'BLEU-{i}': [] for i in range(1, 5)}
    all_meteor_scores = []
    
    # Generate captions for each test image
    for i, (img_stem, image_features) in enumerate(test_features.items()):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(test_features)}")
        
        # Find corresponding caption file name
        img_name = None
        for name in test_captions.keys():
            if Path(name).stem == img_stem:
                img_name = name
                break
        
        if img_name is None:
            continue
        
        # Generate caption using specified method
        if method == 'beam':
            generated_caption = generate_caption_beam(model, image_features, tokenizer, beam_size)
            attn_arr = None
        else:
            if save_attention:
                generated_caption, attn_arr = generate_caption_greedy_with_attention(model, image_features, tokenizer)
                if attn_arr is not None:
                    np.save(Path(attention_dir) / f"{Path(img_name).stem}_attention.npy", attn_arr)
            else:
                generated_caption = generate_caption_greedy(model, image_features, tokenizer)
                attn_arr = None
        generated_captions[img_name] = generated_caption
        
        # Get reference captions
        reference_captions = test_captions[img_name]
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu_scores(reference_captions, generated_caption)
        for metric, score in bleu_scores.items():
            all_bleu_scores[metric].append(score)
        
        # Calculate METEOR score
        meteor_score = calculate_meteor_score(reference_captions, generated_caption)
        all_meteor_scores.append(meteor_score)
    
    # Calculate average scores
    avg_scores = {}
    for metric, scores in all_bleu_scores.items():
        avg_scores[metric] = np.mean(scores)
    
    avg_scores['METEOR'] = np.mean(all_meteor_scores)
    
    # Calculate CIDEr score
    if COCO_EVAL_AVAILABLE:
        # Prepare data for CIDEr calculation
        references_for_cider = {}
        candidates_for_cider = {}
        
        for img_name, ref_captions in test_captions.items():
            if img_name in generated_captions:
                references_for_cider[img_name] = ref_captions
                candidates_for_cider[img_name] = [generated_captions[img_name]]
        
        cider_score = calculate_cider_score(references_for_cider, candidates_for_cider)
        avg_scores['CIDEr'] = cider_score
    
    return avg_scores, generated_captions


def compare_evaluation_methods(model: ImageCaptioningModel, test_features: Dict[str, np.ndarray], 
                             test_captions: Dict[str, List[str]], tokenizer: Tokenizer, 
                             beam_size: int = 3, max_images: int = 100) -> None:
    """
    Compare greedy vs beam search evaluation methods
    """
    print(f"\nComparing evaluation methods on {max_images} images...")
    
    # Limit to subset for comparison
    limited_features = dict(list(test_features.items())[:max_images])
    
    # Evaluate with greedy search
    print("\nEvaluating with Greedy Search...")
    greedy_scores, greedy_captions = evaluate_model(model, limited_features, test_captions, 
                                                   tokenizer, 'greedy')
    
    # Evaluate with beam search
    print("\nEvaluating with Beam Search...")
    beam_scores, beam_captions = evaluate_model(model, limited_features, test_captions, 
                                               tokenizer, 'beam', beam_size)
    
    # Print comparison
    print("\n" + "="*80)
    print("EVALUATION METHOD COMPARISON")
    print("="*80)
    print(f"Number of test images: {max_images}")
    print(f"Beam size: {beam_size}")
    print("-" * 80)
    
    print("BLEU Scores:")
    for i in range(1, 5):
        metric = f'BLEU-{i}'
        greedy_score = greedy_scores.get(metric, 0)
        beam_score = beam_scores.get(metric, 0)
        improvement = beam_score - greedy_score
        print(f"  {metric}:")
        print(f"    Greedy: {greedy_score:.4f}")
        print(f"    Beam:   {beam_score:.4f}")
        print(f"    Δ:      {improvement:+.4f}")
    
    print("-" * 80)
    print("METEOR Scores:")
    greedy_meteor = greedy_scores.get('METEOR', 0)
    beam_meteor = beam_scores.get('METEOR', 0)
    meteor_improvement = beam_meteor - greedy_meteor
    print(f"  Greedy: {greedy_meteor:.4f}")
    print(f"  Beam:   {beam_meteor:.4f}")
    print(f"  Δ:      {meteor_improvement:+.4f}")
    
    print("="*80)
    
    # Save comparison results
    comparison_results = {
        'greedy_scores': greedy_scores,
        'beam_scores': beam_scores,
        'greedy_captions': greedy_captions,
        'beam_captions': beam_captions
    }
    
    with open("evaluation_results/method_comparison.pkl", "wb") as f:
        pickle.dump(comparison_results, f)
    
    print("Comparison results saved to evaluation_results/method_comparison.pkl")


def compare_models_and_save(
    baseline_model_path: str,
    attention_model_path: str,
    test_features: Dict[str, np.ndarray],
    test_captions: Dict[str, List[str]],
    tokenizer: Tokenizer,
    beam_size: int,
    output_json: str = "evaluation_results/metrics_comparison.json"
) -> None:
    """Evaluate baseline and attention models with greedy and beam and save JSON metrics."""
    results = {}

    # Baseline
    print("\nEvaluating BASELINE model...")
    baseline_model = load_trained_model(baseline_model_path, 'baseline')
    scores_baseline_greedy, _ = evaluate_model(baseline_model, test_features, test_captions, tokenizer, 'greedy', beam_size)
    scores_baseline_beam, _ = evaluate_model(baseline_model, test_features, test_captions, tokenizer, 'beam', beam_size)
    results['baseline_greedy'] = scores_baseline_greedy
    results['baseline_beam'] = scores_baseline_beam

    # Attention
    print("\nEvaluating ATTENTION model...")
    attention_model = load_trained_model(attention_model_path, 'attention')
    scores_attention_greedy, _ = evaluate_model(attention_model, test_features, test_captions, tokenizer, 'greedy', beam_size)
    scores_attention_beam, _ = evaluate_model(attention_model, test_features, test_captions, tokenizer, 'beam', beam_size)
    results['attention_greedy'] = scores_attention_greedy
    results['attention_beam'] = scores_attention_beam

    # Save JSON
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics comparison saved to {out_path}")


def print_evaluation_summary(scores: Dict[str, float], num_images: int, method: str = "Greedy"):
    """
    Print a summary of evaluation metrics
    """
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY - {method.upper()} SEARCH")
    print("="*60)
    print(f"Number of test images evaluated: {num_images}")
    print("-" * 60)
    
    # BLEU scores
    print("BLEU Scores:")
    for i in range(1, 5):
        metric = f'BLEU-{i}'
        if metric in scores:
            print(f"  {metric}: {scores[metric]:.4f}")
    
    print("-" * 60)
    
    # METEOR score
    if 'METEOR' in scores:
        print(f"METEOR Score: {scores['METEOR']:.4f}")
    
    # CIDEr score
    if 'CIDEr' in scores:
        print(f"CIDEr Score: {scores['CIDEr']:.4f}")
    
    print("="*60)


def save_results(scores: Dict[str, float], generated_captions: Dict[str, str], output_dir: str):
    """
    Save evaluation results to files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save scores
    with open(output_path / "evaluation_scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    
    # Save generated captions
    with open(output_path / "generated_captions.pkl", "wb") as f:
        pickle.dump(generated_captions, f)
    
    # Save scores as text file
    with open(output_path / "evaluation_results.txt", "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        for metric, score in scores.items():
            f.write(f"{metric}: {score:.4f}\n")
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    """
    Main evaluation pipeline
    """
    parser = argparse.ArgumentParser(description="Evaluate CNN+LSTM Image Captioning Model")
    parser.add_argument("--model_path", type=str, default="models/best_model.h5", 
                       help="Path to the trained model (default: models/best_model.h5)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save evaluation results (default: evaluation_results)")
    parser.add_argument("--max_images", type=int, default=None, 
                       help="Maximum number of images to evaluate (default: all)")
    parser.add_argument("--method", type=str, choices=['greedy', 'beam', 'compare'], default='greedy',
                       help="Evaluation method: greedy, beam, or compare both (default: greedy)")
    parser.add_argument("--beam_size", type=int, default=3,
                       help="Beam size for beam search (default: 3)")
    parser.add_argument("--model_type", type=str, choices=['baseline', 'attention'], default='baseline',
                       help="Model type: baseline or attention (default: baseline)")
    parser.add_argument("--save_attention", action="store_true",
                       help="Save attention maps during greedy decoding (attention model only)")
    parser.add_argument("--attention_dir", type=str, default="evaluation_results/attention_maps",
                       help="Directory to save attention maps")
    parser.add_argument("--compare_models", action="store_true",
                       help="Compare baseline and attention models and save metrics_comparison.json")
    parser.add_argument("--baseline_model_path", type=str, default="models/best_model.h5",
                       help="Path to baseline model (default: models/best_model.h5)")
    parser.add_argument("--attention_model_path", type=str, default="models/best_model_attention.h5",
                       help="Path to attention model (default: models/best_model_attention.h5)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CNN+LSTM IMAGE CAPTIONING MODEL EVALUATION")
    print("="*60)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Load test data
    test_features, test_captions, tokenizer = load_test_data()
    
    # Limit number of images if specified
    if args.max_images:
        test_features = dict(list(test_features.items())[:args.max_images])
        print(f"Limited evaluation to {len(test_features)} images")
    
    # Compare models branch
    if args.compare_models:
        compare_models_and_save(
            baseline_model_path=args.baseline_model_path,
            attention_model_path=args.attention_model_path,
            test_features=test_features,
            test_captions=test_captions,
            tokenizer=tokenizer,
            beam_size=args.beam_size,
            output_json="evaluation_results/metrics_comparison.json"
        )
        print("Evaluation completed!")
        return

    # Load trained model
    model = load_trained_model(args.model_path, args.model_type)
    
    if args.method == 'compare':
        # Compare both methods
        compare_evaluation_methods(model, test_features, test_captions, tokenizer, 
                                 args.beam_size, args.max_images or 100)
    else:
        # Single method evaluation
        scores, generated_captions = evaluate_model(
            model, test_features, test_captions, tokenizer,
            args.method, args.beam_size,
            save_attention=args.save_attention,
            attention_dir=args.attention_dir
        )
        
        # Print summary
        print_evaluation_summary(scores, len(test_features), args.method)
        
        # Save results
        save_results(scores, generated_captions, args.output_dir)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
