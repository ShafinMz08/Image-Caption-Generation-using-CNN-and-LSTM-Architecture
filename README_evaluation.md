# Image Captioning Model Evaluation

This document describes the evaluation results for the CNN+LSTM Image Captioning Model.

## Model Performance

The model was evaluated on the Flickr8k test dataset using multiple metrics:

### Evaluation Metrics (100 test images)

- **BLEU-1**: 0.3251
- **BLEU-2**: 0.1413
- **BLEU-3**: 0.0728
- **BLEU-4**: 0.0510
- **METEOR**: 0.1785

### Model Architecture

- **CNN**: ResNet50 (pre-trained, frozen)
- **LSTM**: 512 units
- **Embedding**: 256 dimensions
- **Vocabulary**: 8,779 words
- **Max sequence length**: 34 tokens

### Training Details

- **Training epochs**: 42 (early stopping at epoch 37)
- **Best validation loss**: 3.61975
- **Final training accuracy**: 38.12%
- **Final validation accuracy**: 36.15%
- **Test accuracy**: 36.11%

## Usage

### Running Evaluation

```bash
# Evaluate on all test images
python evaluate.py

# Evaluate on a subset of images
python evaluate.py --max_images 100

# Use a different model
python evaluate.py --model_path models/final_model.h5
```

### Viewing Results

```bash
# Show example generated captions
python show_examples.py
```

## Files Generated

- `evaluation_results/evaluation_scores.pkl` - Pickle file with all scores
- `evaluation_results/generated_captions.pkl` - Generated captions for all test images
- `evaluation_results/evaluation_results.txt` - Human-readable results

## Notes

1. **CIDEr scores** are not calculated as the `pycocoevalcap` library is not installed
2. **Model behavior**: The model tends to generate similar captions, suggesting it may need more training or architectural improvements
3. **BLEU scores** are reasonable for a basic model but could be improved with:
   - Beam search instead of greedy search
   - Attention mechanisms
   - More training data
   - Better hyperparameter tuning

## Dependencies

- TensorFlow/Keras
- NLTK (for BLEU and METEOR scores)
- NumPy
- Pickle

## Installation

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```
