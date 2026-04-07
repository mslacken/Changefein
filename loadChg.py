from datasets import load_dataset
from jinja2 import Template
from transformers import AutoTokenizer

MAX_LENGTH = 1024

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

# Jinja2 template for formatting the changelog entry
CHANGELOG_TEMPLATE = (
    '''{% if package %}create structured changelog for package {{ package }}{% endif %} {% if version %} {{ version }}{% endif %}:
{% if archive_changelog %}changelog:
{{ archive_changelog }}{% endif %}
{% if github_release_notes %}release notes:
{{ github_release_notes }}{% endif %}
{% if added_files %}new files: {{ added_files }}{% endif %}
{% if removed_files %}removed files: {{ removed_files }}{% endif %}
{% if spec_diff %}changes in spec file:
{{ spec_diff }}{% endif %}'''
)
template = Template(CHANGELOG_TEMPLATE)

def preprocess_function(data):
    """
    Format the changelog for T5 input using Jinja2.
    Ensures the output doesn't exceed MAX_LENGTH after tokenization.
    Fields 'package', 'version', 'added_files', 'removed_files' stay intact.
    Others ('archive_changelog', 'github_release_notes', 'spec_diff') are truncated if necessary.
    """
    batch_size = len(next(iter(data.values())))
    optional_keys = ['archive_changelog', 'github_release_notes', 'spec_diff']
    
    results = []
    for i in range(batch_size):
        # Extract fields for the current entry in the batch
        item = {field: data[field][i] for field in data if i < len(data[field]) and data[field][i]}
        
        # Pre-tokenize the optional fields to allow truncation
        tokenized_optional = {}
        for k in optional_keys:
            if k in item and item[k]:
                tokenized_optional[k] = tokenizer.encode(item[k], add_special_tokens=False)

        def get_rendered_text(L):
            temp_item = dict(item)
            for k in optional_keys:
                if k in temp_item and temp_item[k]:
                    # Truncate tokens and decode back to text
                    truncated_tokens = tokenized_optional[k][:L]
                    temp_item[k] = tokenizer.decode(truncated_tokens)
            
            rendered = template.render(**temp_item)
            # Filter out empty lines
            return "\n".join(line for line in rendered.splitlines() if line.strip())

        # Check if full text fits without truncation
        # Using a very large value for L effectively means no truncation here
        # But for robustness, we use a reasonable upper bound
        full_text = get_rendered_text(2**31) # Effectively infinity
        if len(tokenizer.encode(full_text)) <= MAX_LENGTH:
            results.append(full_text)
            continue
            
        # Binary search for the optimal truncation length L for optional fields
        low, high = 0, MAX_LENGTH
        best_text = get_rendered_text(0)
        
        while low <= high:
            mid = (low + high) // 2
            current_text = get_rendered_text(mid)
            if len(tokenizer.encode(current_text)) <= MAX_LENGTH:
                best_text = current_text
                low = mid + 1
            else:
                high = mid - 1
                
        results.append(best_text)
    
    return results
        

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
