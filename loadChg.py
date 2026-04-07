from datasets import load_dataset
from jinja2 import Template

MAX_LENGTH = 1024

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
    Assuming data is a batch (dict of lists) as typically passed to dataset.map(batched=True).
    """
    # Determine batch size from any available field in the data
    batch_size = len(next(iter(data.values())))
    
    results = []
    for i in range(batch_size):
        # Extract fields for the current entry in the batch
        item = {field: data[field][i] for field in data if i < len(data[field]) and data[field][i]}
        # Render the template and strip any surrounding whitespace
        rendered_text = template.render(**item)
        # Filter out empty lines
        filtered_text = "\n".join(line for line in rendered_text.splitlines() if line.strip())
        results.append(filtered_text)
    
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
