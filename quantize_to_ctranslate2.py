#!/usr/bin/env python3
"""
Convert and quantize a fine-tuned T5 model to CTranslate2 format.

CTranslate2 is a fast inference engine for Transformer models with
built-in quantization support. It's the T5 equivalent of llama.cpp.
"""

import argparse
import os
import sys
from pathlib import Path


# Quantization types supported by CTranslate2
QUANT_TYPES = [
    'int8',        # 8-bit integer quantization (recommended, ~4x smaller)
    'int8_float32', # INT8 weights, FP32 activations (balanced)
    'int8_float16', # INT8 weights, FP16 activations (faster on GPU)
    'int8_bfloat16', # INT8 weights, BF16 activations
    'int16',       # 16-bit integer quantization
    'float16',     # 16-bit float (2x smaller, GPU optimized)
    'bfloat16',    # Brain float 16
    'float32',     # 32-bit float (no quantization)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert T5 model to CTranslate2 format with quantization"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./results_t5-small_finetune",
        help="Directory containing the fine-tuned model (default: ./results_t5-small_finetune)"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="int8",
        choices=QUANT_TYPES,
        help=f"Quantization type (default: int8). Options: {', '.join(QUANT_TYPES)}"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for CTranslate2 files (default: {model_dir}/ctranslate2)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory"
    )
    return parser.parse_args()


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import ctranslate2
    except ImportError:
        missing.append("ctranslate2")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    if missing:
        print("Error: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def convert_to_ctranslate2(model_dir, out_dir, quant_type, force):
    """Convert HuggingFace T5 model to CTranslate2 format."""
    import ctranslate2
    import json
    import shutil
    import tempfile

    out_path = Path(out_dir)

    # Check if output directory exists
    if out_path.exists() and not force:
        print(f"\nError: Output directory already exists: {out_path}")
        print("Use --force to overwrite")
        return None

    print(f"\nConverting {model_dir} to CTranslate2 format...")
    print(f"Quantization: {quant_type}")
    print(f"Output: {out_path}")

    try:
        # Workaround for tokenizer config issue with extra_special_tokens
        # Create a temporary directory with fixed tokenizer config
        model_path = Path(model_dir)
        tokenizer_config_path = model_path / "tokenizer_config.json"

        needs_temp_fix = False
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)

            # Check if extra_special_tokens is a list (causes CTranslate2 error)
            if 'extra_special_tokens' in tokenizer_config:
                if isinstance(tokenizer_config['extra_special_tokens'], list):
                    needs_temp_fix = True

        if needs_temp_fix:
            print("Note: Fixing tokenizer config for CTranslate2 compatibility...")
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="ct2_convert_")
            temp_model_path = Path(temp_dir) / "model"

            # Copy model files
            shutil.copytree(model_path, temp_model_path)

            # Fix tokenizer config - remove extra_special_tokens
            temp_tokenizer_config = temp_model_path / "tokenizer_config.json"
            with open(temp_tokenizer_config, 'r') as f:
                config = json.load(f)

            # Remove the problematic field
            config.pop('extra_special_tokens', None)

            with open(temp_tokenizer_config, 'w') as f:
                json.dump(config, f, indent=2)

            conversion_path = str(temp_model_path)
        else:
            temp_dir = None
            conversion_path = str(model_dir)

        # Convert model
        converter = ctranslate2.converters.TransformersConverter(conversion_path)
        converter.convert(
            output_dir=str(out_path),
            quantization=quant_type,
            force=force
        )

        # Clean up temp directory
        if temp_dir:
            shutil.rmtree(temp_dir)

        print("✓ Conversion successful!")
        return out_path

    except Exception as e:
        print(f"\nError: Conversion failed: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp directory on error
        if 'temp_dir' in locals() and temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

        return None


def verify_model(model_dir, ct2_dir):
    """Verify the converted model works."""
    import ctranslate2
    from transformers import T5Tokenizer
    import json
    import tempfile
    import shutil

    print("\nVerifying converted model...")

    try:
        # Load CTranslate2 model
        translator = ctranslate2.Translator(str(ct2_dir))

        # Load tokenizer with workaround for extra_special_tokens issue
        model_path = Path(model_dir)
        tokenizer_config_path = model_path / "tokenizer_config.json"

        # Check if we need to fix the tokenizer
        needs_fix = False
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, 'r') as f:
                config = json.load(f)
            if 'extra_special_tokens' in config and isinstance(config['extra_special_tokens'], list):
                needs_fix = True

        if needs_fix:
            # Create temporary fixed tokenizer
            temp_dir = tempfile.mkdtemp(prefix="ct2_verify_")
            temp_model_path = Path(temp_dir) / "model"
            shutil.copytree(model_path, temp_model_path)

            # Fix config
            temp_config_path = temp_model_path / "tokenizer_config.json"
            with open(temp_config_path, 'r') as f:
                config = json.load(f)
            config.pop('extra_special_tokens', None)
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)

            tokenizer = T5Tokenizer.from_pretrained(str(temp_model_path))
            shutil.rmtree(temp_dir)
        else:
            tokenizer = T5Tokenizer.from_pretrained(str(model_path))

        # Test input
        test_input = "generate changelog: test"
        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_input))

        # Generate
        results = translator.translate_batch([input_tokens])

        # Decode
        output_tokens = results[0].hypotheses[0]
        output_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True
        )

        print(f"✓ Model verification successful!")
        print(f"  Test input: '{test_input}'")
        print(f"  Test output: '{output_text}'")

        return True

    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("\nMake sure you've trained a model first using train.py")
        sys.exit(1)

    # Check for required files
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"Error: config.json not found in {model_dir}")
        print("This doesn't appear to be a valid model directory")
        sys.exit(1)

    # Set output directory
    out_dir = args.out_dir or str(model_dir / "ctranslate2")

    print("=" * 60)
    print("T5 Model CTranslate2 Conversion")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Quantization type: {args.quant_type}")
    print(f"Output directory: {out_dir}")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Get original model size
    print("\nCalculating original model size...")
    original_size = get_directory_size(model_dir)
    print(f"Original model size: {original_size / (1024**2):.1f} MB")

    # Convert
    ct2_path = convert_to_ctranslate2(model_dir, out_dir, args.quant_type, args.force)
    if ct2_path is None:
        sys.exit(1)

    # Get converted model size
    ct2_size = get_directory_size(ct2_path)
    compression = (1 - ct2_size / original_size) * 100

    print("\nFile sizes:")
    print(f"  Original: {original_size / (1024**2):.1f} MB")
    print(f"  CTranslate2 ({args.quant_type}): {ct2_size / (1024**2):.1f} MB")
    print(f"  Compression: {compression:.1f}%")

    # Verify model
    if verify_model(model_dir, ct2_path):
        print("\n" + "=" * 60)
        print("✓ Conversion complete!")
        print("=" * 60)
        print(f"CTranslate2 model: {ct2_path}")
        print("\nUsage example:")
        print("  import ctranslate2")
        print("  from transformers import AutoTokenizer")
        print(f"  translator = ctranslate2.Translator('{ct2_path}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{model_dir}')")
        print("\nOr use the CLI:")
        print(f"  python t5-cli.py '{ct2_path}' '{model_dir}' 'your prompt here'")
    else:
        print("\nWarning: Model conversion completed but verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
