import torch
import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from preprocess import preprocess_function

def main():
    # Configuration
    # Usage: python test_t5.py [model_path_or_id] [n_samples]
    
    MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google-t5/t5-small"
    
    try:
        n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    except ValueError:
        print("Invalid sample count. Please provide an integer as the second argument.")
        sys.exit(1)

    print(f"Loading model and tokenizer from: {MODEL_ID}...")
    
    # Check if path exists locally
    if os.path.isdir(MODEL_ID):
        print(f"Loading local model from {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model = model.to(torch.float32) # Ensure float32 on CPU to avoid some op errors
    model.to(device)

    print(f"Loading dataset from changes.json...")
    # Load dataset and select samples
    dataset = load_dataset('json', data_files="changes.json")['train']
    
    # We use a seed for reproducibility when testing
    dataset = dataset.shuffle(seed=42)
    samples = dataset.select(range(min(n_samples, len(dataset)))).to_dict()

    print(f"Preprocessing {n_samples} samples...")
    # Preprocess samples
    preprocessed_texts = preprocess_function(samples)

    print(f"\nTesting with model: {MODEL_ID}")
    print("=" * 60)

    for i, text in enumerate(preprocessed_texts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input text (truncated): {text[:200]}...")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        # Generate output
        with torch.no_grad():
            output = model.generate(inputs.input_ids, max_new_tokens=512)
        
        # Decode and print output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"\nTarget (Expected):")
        print(samples['changes_diff'][i])
        
        print(f"\nGenerated Output:")
        print(decoded_output)
        print("-" * 60)

if __name__ == "__main__":
    main()
