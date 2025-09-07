import pickle
from pathlib import Path

def show_caption_examples():
    """
    Show some example generated captions
    """
    results_path = Path("evaluation_results")
    
    if not results_path.exists():
        print("No evaluation results found. Please run evaluate.py first.")
        return
    
    # Load generated captions
    with open(results_path / "generated_captions.pkl", "rb") as f:
        generated_captions = pickle.load(f)
    
    # Load test data to get reference captions
    with open("data/processed/splits/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    print("="*80)
    print("EXAMPLE GENERATED CAPTIONS")
    print("="*80)
    
    # Show first 5 examples
    count = 0
    for img_name, generated_caption in generated_captions.items():
        if count >= 5:
            break
            
        # Get reference captions
        reference_captions = test_data['captions'].get(img_name, [])
        
        print(f"\nImage: {img_name}")
        print("-" * 50)
        print("Generated Caption:")
        print(f"  {generated_caption}")
        print("\nReference Captions:")
        for i, ref_caption in enumerate(reference_captions, 1):
            # Clean up reference caption
            clean_ref = ref_caption.replace('<start>', '').replace('<end>', '').strip()
            print(f"  {i}. {clean_ref}")
        
        count += 1
    
    print("\n" + "="*80)

if __name__ == "__main__":
    show_caption_examples()
