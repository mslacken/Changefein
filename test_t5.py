import argparse
import torch
import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from preprocess import preprocess_function

def parse_args():
    parser = argparse.ArgumentParser(description="Test a T5 model on changelog data")
    parser.add_argument(
        "--model", 
        type=str, 
        default="google-t5/t5-small",
        help="The model ID or local path to use (default: google-t5/t5-small)"
    )
    parser.add_argument(
        "--samples", 
        "-n",
        type=int, 
        default=3,
        help="Number of samples to test (default: 3)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sample selection (default: 42)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    MODEL_ID = args.model
    n_samples = args.samples
    seed = args.seed

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
    dataset = dataset.shuffle(seed=seed)
    samples = dataset.select(range(min(n_samples, len(dataset)))).to_dict()

    print(f"Preprocessing {n_samples} samples (Seed: {seed})...")
    # Preprocess samples
    preprocessed_texts = preprocess_function(samples)

    print("=" * 80)

    for i in range(len(preprocessed_texts)):
        input_str = preprocessed_texts[i]
        target_str = samples['changes_diff'][i]
        
        # Tokenize input and target for stats
        input_tokens = tokenizer.encode(input_str)
        target_tokens = tokenizer.encode(target_str)
        
        print(f"Example {i+1}:")
        print(f"  Input Tokens: {len(input_tokens)}")
        print(f"  Target Tokens: {len(target_tokens)}")
        print("-" * 40)
        
        # Prepare model input
        inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        # Generate output
        with torch.no_grad():
            output = model.generate(inputs.input_ids, max_new_tokens=512)
        
        # Decode output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_tokens = tokenizer.encode(decoded_output)
        
        print("INPUT (Formatted):")
        print(input_str)
        print("-" * 20)
        print("TARGET (Expected):")
        print(target_str)
        print("-" * 20)
        print(f"GENERATED OUTPUT ({len(generated_tokens)} tokens):")
        print(decoded_output)
        print("=" * 80)

if __name__ == "__main__":
    main()
