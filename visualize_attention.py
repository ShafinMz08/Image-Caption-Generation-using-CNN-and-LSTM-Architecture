import argparse
from pathlib import Path
import numpy as np
from typing import Optional

from PIL import Image
import matplotlib.pyplot as plt

from src.config import CONFIG
from src.dataset import load_tokenizer
from predict import load_trained_model, predict_single_image


def overlay_attention_heatmap(image_path: Path, attention_vector: np.ndarray, out_path: Path, title: str = ""):
    """
    Save an attention heatmap overlay on the image. Our attention shape is (1,1) per step, so
    we visualize it as a scalar bar. This is a simplified visualization due to non-spatial features.
    """
    attention_score = float(attention_vector.squeeze()) if attention_vector is not None else 0.0
    img = Image.open(image_path).convert('RGB')

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Image')

    # Right: simple bar for attention strength
    ax[1].imshow(np.ones((224, 50, 3)))
    ax[1].axis('off')
    ax[1].set_title(f'Attention: {attention_score:.3f}')
    plt.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_attention(
    model_type: str,
    model_path: str,
    image_path: str,
    method: str = 'greedy',
    beam_size: int = 3,
    attention_dir: str = 'evaluation_results/visualizations'
):
    tokenizer = load_tokenizer(CONFIG.TOKENIZER_PATH)
    model = load_trained_model(model_path, model_type)

    # Generate caption and collect/save attention arrays via predict_single_image helper
    image_stem = Path(image_path).stem
    out_dir = Path(attention_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    caption = predict_single_image(
        model, image_path, tokenizer, method=method, beam_size=beam_size,
        save_attention=True, attention_dir=str(out_dir)
    )

    # Load saved attention array if available
    attn_path = out_dir / f"{image_stem}_attention.npy"
    if attn_path.exists():
        attn = np.load(attn_path, allow_pickle=True)
        # attn expected shape: (steps, 1, 1) in our simplified attention
        # Render a simple strip of attention scalar plots
        for t in range(min(attn.shape[0], 20)):
            overlay_attention_heatmap(Path(image_path), attn[t], out_dir / f"{image_stem}_step{t:02d}.png", title=caption)
        print(f"✓ Saved attention visualizations to {out_dir}")
    else:
        print("Warning: No attention file found; ensure model_type=attention and --save_attention path is correct.")

    # Save caption text
    with open(out_dir / f"{image_stem}_caption.txt", 'w', encoding='utf-8') as f:
        f.write(caption + "\n")
    print(f"Caption: {caption}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention for image captioning')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'attention'], default='attention')
    parser.add_argument('--model_path', type=str, default='models/best_model_attention.h5')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='evaluation_results/visualizations')

    args = parser.parse_args()
    visualize_attention(args.model_type, args.model_path, args.image_path, args.method, args.beam_size, args.output_dir)


if __name__ == '__main__':
    main()


