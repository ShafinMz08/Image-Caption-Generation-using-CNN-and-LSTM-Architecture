# Beam Search Implementation for Image Captioning

This document describes the beam search implementation for the CNN+LSTM Image Captioning Model.

## Implementation Details

### Beam Search Algorithm

The beam search implementation includes:

1. **BeamSearchNode Class**: Represents nodes in the search tree with:

   - `sequence`: List of token IDs
   - `log_prob`: Cumulative log probability
   - `is_finished`: Flag for completed sequences

2. **Beam Search Process**:

   - Maintains top-k sequences (beam_size) at each timestep
   - Computes cumulative log probabilities for candidate sequences
   - Stops when `<end>` token is reached or max length exceeded
   - Returns the sequence with highest probability

3. **Key Features**:
   - Configurable beam size (default: 3)
   - Proper handling of start/end tokens
   - Fallback to greedy search if no sequences finish
   - Efficient heap-based candidate selection

## Usage

### Prediction Script (predict.py)

```bash
# Generate captions with greedy search
python predict.py --method greedy --test_subset 10

# Generate captions with beam search
python predict.py --method beam --beam_size 5 --test_subset 10

# Compare both methods
python predict.py --compare --beam_size 3 --test_subset 10

# Single image prediction
python predict.py --image_path path/to/image.jpg --method beam --beam_size 3
```

### Evaluation Script (evaluate.py)

```bash
# Evaluate with greedy search
python evaluate.py --method greedy --max_images 100

# Evaluate with beam search
python evaluate.py --method beam --beam_size 3 --max_images 100

# Compare both methods
python evaluate.py --method compare --beam_size 3 --max_images 50
```

## Results Comparison

### Test Results (20 images, beam_size=3)

| Metric | Greedy | Beam   | Δ (Beam - Greedy) |
| ------ | ------ | ------ | ----------------- |
| BLEU-1 | 0.3531 | 0.0830 | -0.2702           |
| BLEU-2 | 0.1600 | 0.0303 | -0.1297           |
| BLEU-3 | 0.0776 | 0.0248 | -0.0528           |
| BLEU-4 | 0.0527 | 0.0267 | -0.0260           |
| METEOR | 0.2081 | 0.0541 | -0.1540           |

### Analysis

**Greedy search outperforms beam search** in this case, which indicates:

1. **Model Limitations**: The model may not be well-trained enough for beam search to be beneficial
2. **Repetitive Generation**: Both methods generate similar, repetitive captions
3. **Training Issues**: The model appears to have learned a limited vocabulary and patterns

### Why Greedy Might Be Better Here

1. **Overfitting**: The model may have overfitted to common patterns
2. **Limited Diversity**: Beam search explores more diverse paths, but the model's probability distribution may be too peaked
3. **Training Data**: The model might not have seen enough diverse caption patterns during training

## Recommendations

1. **Improve Training**:

   - Train for more epochs
   - Use better regularization
   - Implement attention mechanisms
   - Use larger, more diverse datasets

2. **Model Architecture**:

   - Add attention layers
   - Use transformer-based architectures
   - Implement better sequence-to-sequence models

3. **Beam Search Tuning**:
   - Try different beam sizes (1, 5, 10)
   - Implement length normalization
   - Add diversity penalties
   - Use nucleus sampling or top-k sampling

## Files Created

- `predict.py`: Main prediction script with beam search
- `evaluate.py`: Updated evaluation script with comparison
- `README_beam_search.md`: This documentation
- `evaluation_results/method_comparison.pkl`: Comparison results

## Dependencies

- TensorFlow/Keras
- NumPy
- NLTK (for evaluation metrics)
- PIL (for image processing)
- Pickle

## Future Improvements

1. **Advanced Decoding**:

   - Nucleus sampling (top-p)
   - Temperature scaling
   - Length normalization for beam search

2. **Model Improvements**:

   - Attention mechanisms
   - Better feature extraction
   - Ensemble methods

3. **Evaluation**:
   - Human evaluation
   - More diverse test sets
   - Qualitative analysis of generated captions
