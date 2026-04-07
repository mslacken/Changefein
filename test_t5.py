import torch
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from preprocess import preprocess_function

def main():
    # Configuration
    try:
        n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    except ValueError:
        print("Invalid sample count. Please provide an integer.")
        sys.exit(1)
        
    MODEL_ID = "google-t5/t5-small"

    print(f"Loading model and tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cpu"
    )

    print(f"Loading dataset from changes.json...")
    # Load dataset and select samples
    dataset = load_dataset('json', data_files="changes.json")['train']
    samples = dataset.select(range(min(n_samples, len(dataset)))).to_dict()

    print(f"Preprocessing {n_samples} samples...")
    # Preprocess samples
    preprocessed_texts = preprocess_function(samples)

    for i, text in enumerate(preprocessed_texts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input text: {text}")
        
        # Tokenize input
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        
        # Generate output
        output = model.generate(input_ids, max_new_tokens=1024)
        
        # Decode and print output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Output: {decoded_output}")

if __name__ == "__main__":
    main()
