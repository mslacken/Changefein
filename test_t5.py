import argparse
import torch
import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from preprocess import preprocess_function, preprocess_target_function

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
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="Maximum length for input sequences (default: 512)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use of CPU even if CUDA is available"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    MODEL_ID = args.model
    n_samples = args.samples
    seed = args.seed
    max_input_length = args.max_input_length

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")

    print(f"Loading model and tokenizer from: {MODEL_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.add_special_tokens({'additional_special_tokens': ['\n']})
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
    except torch.OutOfMemoryError:
        print("Error: CUDA out of memory while loading the model. Try running with --cpu.")
        sys.exit(1)
    
    # Resize model embeddings to account for the new '\n' token
    model.resize_token_embeddings(len(tokenizer))
    
    if device == "cpu":
        model = model.to(torch.float32) # Ensure float32 on CPU
    model.to(device)

    print(f"Loading dataset from changes.json...")
    # Load dataset and select samples
    if not os.path.exists("changes.json"):
        print("Error: changes.json not found.")
        sys.exit(1)

    dataset = load_dataset('json', data_files="changes.json")['train']
    
    # We use a seed for reproducibility when testing
    dataset = dataset.shuffle(seed=seed)
    samples = dataset.select(range(min(n_samples, len(dataset)))).to_dict()

    print(f"Preprocessing {n_samples} samples (Seed: {seed})...")
    # Preprocess samples
    preprocessed_texts = preprocess_function(samples, tokenizer, max_length=max_input_length)
    # Preprocess targets
    preprocessed_targets = preprocess_target_function(samples)

    print("=" * 80)

    for i in range(len(preprocessed_texts)):
        input_str = preprocessed_texts[i]
        target_str = preprocessed_targets[i]
        
        # Tokenize input and target for stats
        input_tokens = tokenizer.encode(input_str)
        target_tokens = tokenizer.encode(target_str)
        
        print(f"Example {i+1}:")
        print(f"  Input Tokens: {len(input_tokens)}")
        print(f"  Target Tokens: {len(target_tokens)}")
        print("-" * 40)
        
        # Prepare model input
        inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        
        # Generate output
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids, 
                max_new_tokens=512,
                repetition_penalty=2.5,     # Discourage repetition
                no_repeat_ngram_size=3,     # Prevent 3-word repeating phrases
                early_stopping=True         # Stop at EOS
            )
        
        # Decode output
        # To keep the newlines (which are special tokens), we decode without skipping special tokens
        # and then manually remove the standard T5 special tokens.
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)
        for sp_token in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.unk_token]:
            if sp_token:
                decoded_output = decoded_output.replace(sp_token, "")
        
        generated_tokens = tokenizer.encode(decoded_output)
        
        print("INPUT (Formatted):")
        print(input_str)
        print("-" * 20)
        print("TARGET (Processed):")
        print(target_str)
        print("-" * 20)
        print(f"GENERATED OUTPUT ({len(generated_tokens)} tokens):")
        print(decoded_output.strip())
        print("=" * 80)

if __name__ == "__main__":
    main()
