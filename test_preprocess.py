from datasets import load_dataset
from preprocess import preprocess_function, tokenizer

# Load the dataset from json
dataset = load_dataset('json', data_files="changes.json")['train']
print(f"Dataset keys: {dataset.features.keys()}")

# Split into train and test
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
dtrain, dtest = dataset_split['train'], dataset_split['test']

# Print the first ten sets (items) from the training set formatted as intended
first_ten_batch = dtrain.select(range(min(10, len(dtrain)))).to_dict()
formatted_strings = preprocess_function(first_ten_batch)

print("First ten sets of formatted changelog strings:")
print("=" * 60)
for i, formatted_str in enumerate(formatted_strings, 1):
    num_tokens = len(tokenizer.encode(formatted_str))
    print(f"Set {i} ({num_tokens} tokens):")
    print(formatted_str)
    print("-" * 60)
