#!/usr/bin/env python3
"""
Test a T5 model on changelog data using CTranslate2.

This script works like test_t5.py but uses CTranslate2 for inference,
supporting both quantized and unquantized models.
"""

import argparse
import sys
import os
import re
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a CTranslate2 T5 model on changelog data"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to CTranslate2 model directory"
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=None,
        help="Path to tokenizer (default: google-t5/t5-small to avoid tokenizer config issues)"
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
        "--max_output_length",
        type=int,
        default=512,
        help="Maximum length for output sequences (default: 512)"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size for generation (default: 1 = greedy)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (show timing and statistics)"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path, tokenizer_path, device, verbose):
    """Load CTranslate2 model and tokenizer."""
    try:
        import ctranslate2
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("\nInstall with:")
        print("  pip install ctranslate2 transformers datasets")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Use base T5 tokenizer by default to avoid extra_special_tokens issue
    if tokenizer_path is None:
        tokenizer_path = "google-t5/t5-small"
        if verbose:
            print(f"Using base T5 tokenizer: {tokenizer_path}")

    if verbose:
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")
        print(f"Device: {device}")

    # Load CTranslate2 model
    translator = ctranslate2.Translator(
        str(model_path),
        device=device,
        compute_type="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    if verbose:
        print(f"✓ Model and tokenizer loaded successfully\n")

    return translator, tokenizer


def load_and_prepare_data(dataset_file, n_samples, seed, verbose):
    """Load dataset and select samples."""
    try:
        from datasets import load_dataset
        from preprocess import preprocess_function, preprocess_target_function
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("\nInstall with:")
        print("  pip install datasets")
        sys.exit(1)

    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found.")
        sys.exit(1)

    if verbose:
        print(f"Loading dataset from {dataset_file}...")

    dataset = load_dataset('json', data_files=dataset_file)['train']

    # Shuffle and select samples
    dataset = dataset.shuffle(seed=seed)
    samples = dataset.select(range(min(n_samples, len(dataset)))).to_dict()

    if verbose:
        print(f"Selected {len(samples['changes_diff'])} samples (Seed: {seed})\n")

    return samples


def generate_ctranslate2(translator, tokenizer, input_text, max_length, beam_size, verbose):
    """Generate output using CTranslate2."""
    start_time = time.time()

    # Tokenize input
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

    if verbose:
        print(f"  Input tokens: {len(input_tokens)}")

    # Generate
    results = translator.translate_batch(
        [input_tokens],
        max_batch_size=1,
        beam_size=beam_size,
        max_decoding_length=max_length,
        return_scores=False
    )

    # Decode
    output_tokens = results[0].hypotheses[0]

    # Decode without skipping special tokens to handle newlines
    decoded_output = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(output_tokens),
        skip_special_tokens=False
    )

    # Remove standard special tokens manually (keep newlines)
    special_to_remove = [
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        "<pad>", "</s>", "<unk>"
    ]
    for sp_token in special_to_remove:
        if sp_token:
            decoded_output = decoded_output.replace(sp_token, "")

    # Remove sentinel tokens (extra_id_XX)
    decoded_output = re.sub(r"<extra_id_\d+>", "", decoded_output)

    end_time = time.time()
    inference_time = end_time - start_time

    if verbose:
        print(f"  Output tokens: {len(output_tokens)}")
        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print(f"  Speed: {len(output_tokens) / inference_time:.1f} tokens/sec")

    return decoded_output.strip(), inference_time


def main():
    args = parse_args()

    print("=" * 80)
    print("T5 CTranslate2 Testing")
    print("=" * 80)

    # Load model and tokenizer
    translator, tokenizer = load_model_and_tokenizer(
        args.model,
        args.tokenizer,
        args.device,
        args.verbose
    )

    # Load dataset
    samples = load_and_prepare_data(
        "changes.json",
        args.samples,
        args.seed,
        args.verbose
    )

    # Import preprocessing functions
    from preprocess import preprocess_function, preprocess_target_function

    # Preprocess samples
    if args.verbose:
        print(f"Preprocessing {args.samples} samples...")

    preprocessed_texts = preprocess_function(
        samples,
        tokenizer,
        max_length=args.max_input_length
    )
    preprocessed_targets = preprocess_target_function(samples)

    if args.verbose:
        print()

    print("=" * 80)

    # Track statistics
    total_time = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Test each sample
    for i in range(len(preprocessed_texts)):
        input_str = preprocessed_texts[i]
        target_str = preprocessed_targets[i]

        # Tokenize input and target for stats
        input_tokens = tokenizer.encode(input_str)
        target_tokens = tokenizer.encode(target_str)

        total_input_tokens += len(input_tokens)

        print(f"Example {i+1}:")
        print(f"  Input Tokens: {len(input_tokens)}")
        print(f"  Target Tokens: {len(target_tokens)}")
        print("-" * 40)

        # Generate output
        generated_output, inference_time = generate_ctranslate2(
            translator,
            tokenizer,
            input_str,
            args.max_output_length,
            args.beam_size,
            args.verbose
        )

        total_time += inference_time

        generated_tokens = tokenizer.encode(generated_output)
        total_output_tokens += len(generated_tokens)

        # Display results
        print("INPUT (Formatted):")
        print(input_str)
        print("-" * 20)
        print("TARGET (Processed):")
        print(target_str)
        print("-" * 20)
        print(f"GENERATED OUTPUT ({len(generated_tokens)} tokens):")
        print(generated_output)

        if not args.verbose:
            print(f"  ({inference_time*1000:.1f}ms)")

        print("=" * 80)

    # Print summary statistics
    if args.verbose and args.samples > 1:
        print("\nSummary Statistics:")
        print(f"  Total samples: {args.samples}")
        print(f"  Total input tokens: {total_input_tokens}")
        print(f"  Total output tokens: {total_output_tokens}")
        print(f"  Total inference time: {total_time*1000:.1f}ms")
        print(f"  Average time per sample: {total_time/args.samples*1000:.1f}ms")
        print(f"  Average speed: {total_output_tokens / total_time:.1f} tokens/sec")
        print("=" * 80)


if __name__ == "__main__":
    main()

# Usage Examples:
#
# Test with quantized model (int8):
#   python test_t5_ctranslate2.py -m results_t5-small_finetune/ctranslate2 -n 5 -v
#
# Test with specific tokenizer:
#   python test_t5_ctranslate2.py -m results_t5-small_finetune/ctranslate2 \
#                                 -t google-t5/t5-small -n 3
#
# Test with beam search:
#   python test_t5_ctranslate2.py -m results_t5-small_finetune/ctranslate2 \
#                                 --beam_size 4 -n 5
#
# Quick test with different seed:
#   python test_t5_ctranslate2.py -m results_t5-small_finetune/ctranslate2 \
#                                 --seed 123 -n 2
