from jinja2 import Template
from transformers import AutoTokenizer

MAX_LENGTH = 1024


# Jinja2 template for formatting the changelog entry
CHANGELOG_TEMPLATE = (
'''{% if package %}create structured changelog for package {{ package }}{% endif %}{% if version %} {{ version }}{% endif %}:
{% if archive_changelog %}changelog:
{{ archive_changelog }}{% endif %}
{% if github_release_notes %}github release notes:
{{ github_release_notes }}{% endif %}
{% if added_files %}new files: {{ added_files }}{% endif %}
{% if removed_files %}removed files: {{ removed_files }}{% endif %}
{% if spec_diff %}changes in spec file:
{{ spec_diff }}{% endif %}'''
)
template = Template(CHANGELOG_TEMPLATE)

def preprocess_function(data, tokenizer):
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
                    temp_item[k] = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            
            rendered = template.render(**temp_item)
            # Filter out empty lines to save tokens
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

def preprocess_target_function(data):
    """
    Preprocess the target output in the field 'changes_diff'.
    Removes the leading '+' sign from diff lines.
    New lines are preserved so they can be explicitly tokenized, but empty lines are filtered out.
    """
    batch_size = len(next(iter(data.values())))
    results = []
    
    for i in range(batch_size):
        diff = data.get('changes_diff', [])[i]
        if not diff:
            results.append("")
            continue
            
        processed_lines = []
        for line in diff.split("\n"):
            if line.startswith("+"):
                # Remove the leading + sign as it was a diff, ignore the rest of lines
                content = line[1:]
                if content.strip():
                    processed_lines.append(content)
                
        # Join with newline character
        results.append("\n".join(processed_lines))
        
    return results
