from datasets import load_dataset

MAX_LENGTH = 1024

def preprocess_function(data):
    """
    Format the changelog for T5 input.
    Assuming data is a batch (dict of lists) as typically passed to dataset.map(batched=True).
    """
    # Mapping of fields to their specific prefix/format
    field_templates = {
        'package': "create structured changelog for package {val} ",
        'version': "{val}:\n",
        'archive_changelog': "changelog {val}\n",
        'github_release_notes': "github release notes {val}\n",
        'added_files': "new files: {val}\n",
        'removed_files': "removed files: {val}\n",
        'spec_diff': "change in spec file: {val}\n"
    }
    
    # Determine batch size from any available field in the data
    batch_size = len(next(iter(data.values())))
    
    results = []
    for i in range(batch_size):
        entry_parts = []
        for field, template in field_templates.items():
            if field in data:
                val = data[field][i]
                if val: # Only include if non-empty
                    entry_parts.append(template.format(val=val).strip())
        results.append("".join(entry_parts).strip())
    
    return results
        

# Load the dataset from json
dataset = load_dataset('json', data_files="changes.json")['train']
print(f"Dataset keys: {dataset.features.keys()}")

# Split into train and test
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
dtrain, dtest = dataset_split['train'], dataset_split['test']

# Print the first ten sets (items) from the training set formatted as intended
first_ten_batch = dtrain.select(range(10)).to_dict()
formatted_strings = preprocess_function(first_ten_batch)

print("First ten sets of formatted changelog strings:")
print("=" * 60)
for i, formatted_str in enumerate(formatted_strings, 1):
    print(f"Set {i}:")
    print(formatted_str)
    print("-" * 60)
