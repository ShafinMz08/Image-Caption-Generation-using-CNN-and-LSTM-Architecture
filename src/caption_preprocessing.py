import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ------------------------------
# Filesystem helpers
# ------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def auto_locate_captions_file(project_root: Path) -> str:
    candidates = [
        project_root / "data/raw/Flickr8k_text/Flickr8k.token.txt",
        project_root / "data/raw/Flickr8k/Flickr8k_text/Flickr8k.token.txt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fallback scan
    found = list(project_root.glob("data/raw/**/Flickr8k.token.txt"))
    if found:
        return str(found[0])
    raise FileNotFoundError("Could not locate Flickr8k.token.txt under data/raw/")


# ------------------------------
# Caption cleaning and loading
# ------------------------------

_clean_re = re.compile(r"[^a-z\s]")


def clean_caption(text: str) -> str:
    """
    Lowercase and remove punctuation, digits, and special characters, condense spaces.
    """
    text = text.lower().strip()
    text = _clean_re.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines are like: 1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing...
            if "\t" in line:
                key, caption = line.split("\t", 1)
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    continue
                key, caption = parts
            img_filename = key.split("#", 1)[0]
            cleaned = clean_caption(caption)
            if not cleaned:
                continue
            image_to_captions.setdefault(img_filename, []).append(cleaned)
    return image_to_captions


def add_tokens(captions: List[str], start_token: str = "startseq", end_token: str = "endseq") -> List[str]:
    return [f"{start_token} {c} {end_token}" for c in captions]


# ------------------------------
# Tokenization and sequences
# ------------------------------

def build_tokenizer(captions: List[str], num_words: int | None = None, oov_token: str = "<unk>") -> Tokenizer:
    tok = Tokenizer(num_words=num_words, oov_token=oov_token)
    tok.fit_on_texts(captions)
    return tok


def captions_to_padded_sequences(
    tokenizer: Tokenizer,
    captions: List[str],
    max_len: int | None = None,
) -> Tuple[np.ndarray, int]:
    seqs = tokenizer.texts_to_sequences(captions)
    if max_len is None:
        max_len = max(len(s) for s in seqs if s) if seqs else 0
    padded = pad_sequences(seqs, maxlen=max_len, padding="post")
    return padded, max_len


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def save_numpy(arr: np.ndarray, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    np.save(path, arr)


# ------------------------------
# Orchestrator
# ------------------------------

def run_caption_preprocessing(
    project_root: str = ".",
    captions_file: str | None = None,
    output_dir: str = "data/processed",
    num_words: int | None = None,
    max_len: int | None = None,
) -> None:
    root = Path(project_root).resolve()
    captions_file = captions_file or auto_locate_captions_file(root)

    # 1) Load dataset captions (image -> captions)
    img2caps = parse_flickr8k_captions(captions_file)

    # Flatten all captions
    raw_caps: List[str] = []
    for caps in img2caps.values():
        raw_caps.extend(caps)

    # 2-3) Clean already done in parsing; wrap with tokens
    token_wrapped = add_tokens(raw_caps, start_token="startseq", end_token="endseq")

    # 4) Tokenizer
    tokenizer = build_tokenizer(token_wrapped, num_words=num_words, oov_token="<unk>")

    # 6) Sequences and padding
    sequences, final_max_len = captions_to_padded_sequences(tokenizer, token_wrapped, max_len=max_len)

    # Paths
    tok_path = str(root / output_dir / "tokenizer.pkl")
    seqs_path = str(root / output_dir / "captions_padded.npy")

    # 5) Save tokenizer; 6b) Save sequences
    save_tokenizer(tokenizer, tok_path)
    save_numpy(sequences, seqs_path)

    print(f"Tokenizer saved to: {tok_path}")
    print(f"Padded sequences saved to: {seqs_path} (shape={sequences.shape}, max_len={final_max_len})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Caption preprocessing: clean, tokenize, and pad.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--captions_file", type=str, default="", help="Path to Flickr8k.token.txt (auto-detected)")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Where to save tokenizer and sequences")
    parser.add_argument("--num_words", type=int, default=None, help="Limit vocabulary size (most frequent)")
    parser.add_argument("--max_len", type=int, default=None, help="Pad/truncate sequences to this length (auto if omitted)")

    args = parser.parse_args()
    run_caption_preprocessing(
        project_root=args.project_root,
        captions_file=args.captions_file or None,
        output_dir=args.output_dir,
        num_words=args.num_words,
        max_len=args.max_len,
    )



