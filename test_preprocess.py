import os
import json
import argparse
from datasets import load_dataset
from preprocess import preprocess_function, preprocess_target_function, tokenizer, MAX_LENGTH

def test_preprocessing(seed=42):
    # Define the data file
    data_file = "changes.json"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please ensure the dataset exists.")
        return

    print(f"Loading dataset from {data_file}...")
    # Load the dataset from json
    try:
        dataset = load_dataset('json', data_files=data_file)['train']
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset loaded with {len(dataset)} examples.")
    print(f"Dataset keys: {list(dataset.features.keys())}")

    # Split into train and test
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    dtrain = dataset_split['train']

    # Test a random selection of ten sets (items) from the training set
    num_to_test = min(10, len(dtrain))
    print(f"\nTesting preprocessing on {num_to_test} random examples (seed={seed})...")
    
    batch = dtrain.shuffle(seed=seed).select(range(num_to_test)).to_dict()
    
    # Process inputs
    formatted_inputs = preprocess_function(batch)
    # Process targets
    formatted_targets = preprocess_target_function(batch)

    print("=" * 80)
    for i in range(num_to_test):
        input_str = formatted_inputs[i]
        target_str = formatted_targets[i]
        
        input_tokens = tokenizer.encode(input_str)
        target_tokens = tokenizer.encode(target_str)
        
        print(f"Example {i+1}:")
        print(f"  Input Tokens: {len(input_tokens)} (Max: {MAX_LENGTH})")
        print(f"  Target Tokens: {len(target_tokens)}")
        print("-" * 40)
        print("INPUT (Formatted):")
        print(input_str)
        print("-" * 20)
        print("TARGET (Processed):")
        print(target_str)
        print("=" * 80)
        
        # Validation
        if len(input_tokens) > MAX_LENGTH:
            print(f"WARNING: Example {i+1} input exceeds MAX_LENGTH!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test preprocessing on a random subset of the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random selection.")
    args = parser.parse_args()
    
    test_preprocessing(seed=args.seed)
