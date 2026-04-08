import json
from datasets import load_dataset
from preprocess import preprocess_function, tokenizer
import statistics

def print_stats(name, data):
    if not data:
        print(f"{name}: No data")
        return
    print(f"{name}:")
    print(f"  Min:    {min(data)}")
    print(f"  Max:    {max(data)}")
    print(f"  Mean:   {statistics.mean(data):.2f}")
    print(f"  Median: {statistics.median(data):.2f}")

def main():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset('json', data_files="changes.json")['train']
    
    # Process the dataset in one batch for simplicity if it fits in memory, 
    # or handle it in batches. Let's do it in one go if possible.
    print("Processing input strings...")
    all_data = dataset.to_dict()
    formatted_inputs = preprocess_function(all_data)
    
    input_chars = []
    input_tokens = []
    target_chars = []
    target_tokens = []
    
    print("Calculating statistics...")
    for i, formatted_input in enumerate(formatted_inputs):
        # Input statistics
        input_chars.append(len(formatted_input))
        input_tokens.append(len(tokenizer.encode(formatted_input)))
        
        # Target statistics (changes_diff)
        target_text = all_data['changes_diff'][i] or ""
        target_chars.append(len(target_text))
        target_tokens.append(len(tokenizer.encode(target_text)))
        
    print("\n" + "="*30)
    print("CHANGELOG STATISTICS")
    print("="*30)
    print_stats("Input Characters", input_chars)
    print_stats("Input Tokens", input_tokens)
    print("-" * 30)
    print_stats("Target (changes_diff) Characters", target_chars)
    print_stats("Target (changes_diff) Tokens", target_tokens)
    print("="*30)

if __name__ == "__main__":
    main()
