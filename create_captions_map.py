import pickle
import re
from pathlib import Path
from typing import Dict, List

def clean_caption(text: str) -> str:
    """Clean caption text"""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"<start> {text} <end>"

def parse_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """Parse Flickr8k captions file"""
    image_to_captions: Dict[str, List[str]] = {}
    
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse line format: image.jpg#0\tcaption text
            if "\t" in line:
                key, caption = line.split("\t", 1)
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    continue
                key, caption = parts
            
            img_filename = key.split("#", 1)[0]
            cleaned = clean_caption(caption)
            if cleaned:
                image_to_captions.setdefault(img_filename, []).append(cleaned)
    
    return image_to_captions

def main():
    # Create captions map
    captions_file = "data/raw/Flickr8k/Flickr8k_text/Flickr8k.token.txt"
    output_file = "data/processed/captions_map.pkl"
    
    print("Creating captions map...")
    captions_map = parse_flickr8k_captions(captions_file)
    
    # Save to pickle
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(captions_map, f)
    
    print(f"Captions map saved to: {output_file}")
    print(f"Total images: {len(captions_map)}")
    print(f"Total captions: {sum(len(caps) for caps in captions_map.values())}")

if __name__ == "__main__":
    main()

