#!/usr/bin/env python3
"""
Quantize a fine-tuned T5 model to GGUF format.

This script converts a Hugging Face T5 model to GGUF format with various quantization options.
Uses the gguf Python package for conversion (no need to clone llama.cpp).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import struct

# Quantization types supported by llama.cpp
QUANT_TYPES = [
    'Q4_0',    # 4-bit, small size, very high quality loss
    'Q4_1',    # 4-bit, larger, lower quality loss
    'Q5_0',    # 5-bit, medium size, low quality loss
    'Q5_1',    # 5-bit, larger, very low quality loss
    'Q8_0',    # 8-bit, large size, extremely low quality loss
    'F16',     # 16-bit float, no quantization
    'F32',     # 32-bit float, no quantization
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a fine-tuned T5 model to GGUF format"
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
        default="Q4_0",
        choices=QUANT_TYPES,
        help=f"Quantization type (default: Q4_0). Options: {', '.join(QUANT_TYPES)}"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for GGUF files (default: {model_dir}/gguf)"
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="Skip conversion step if GGUF file already exists"
    )
    return parser.parse_args()


def find_quantize_binary():
    """Find llama-quantize binary (system or local)."""
    import shutil

    # Check system installation first
    system_quantize = shutil.which('llama-quantize')
    if system_quantize:
        return Path(system_quantize)

    return None


def check_dependencies():
    """Check if required Python packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import gguf
    except ImportError:
        missing.append("gguf")

    if missing:
        print("Error: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def convert_to_gguf(model_dir, out_dir):
    """Convert HuggingFace T5 model to GGUF format."""
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    import gguf
    import numpy as np

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Base GGUF file (F32)
    base_gguf = out_path / "model-f32.gguf"

    print(f"\nConverting {model_dir} to GGUF format...")
    print(f"Output: {base_gguf}")

    try:
        # Load model and tokenizer
        print("Loading model...")
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Create GGUF writer
        print("Creating GGUF file...")
        writer = gguf.GGUFWriter(str(base_gguf), "t5")

        # Add basic metadata
        writer.add_name(model.config.model_type)
        writer.add_architecture()
        writer.add_block_count(model.config.num_layers)
        writer.add_context_length(model.config.n_positions if hasattr(model.config, 'n_positions') else 512)
        writer.add_embedding_length(model.config.d_model)
        writer.add_feed_forward_length(model.config.d_ff)
        writer.add_head_count(model.config.num_heads)

        # Add T5-specific metadata required by llama.cpp
        writer.add_layer_norm_rms_eps(getattr(model.config, 'layer_norm_epsilon', 1e-6))

        # Add key dimension for attention
        if hasattr(model.config, 'd_kv'):
            writer.add_key_length(model.config.d_kv)
            writer.add_value_length(model.config.d_kv)

        # Add T5 relative attention parameters
        if hasattr(model.config, 'relative_attention_num_buckets'):
            # Add as custom key since gguf may not have specific methods for these
            writer.add_uint32(f"{writer.arch}.attention.relative_buckets_count",
                            model.config.relative_attention_num_buckets)

        if hasattr(model.config, 'relative_attention_max_distance'):
            writer.add_uint32(f"{writer.arch}.attention.relative_distance_max",
                            model.config.relative_attention_max_distance)

        # Add vocab size
        writer.add_file_type(0)  # F32

        # Add tokenizer
        vocab_size = len(tokenizer)
        writer.add_token_list([tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)])

        # Convert and add tensors
        print("Converting tensors to GGUF format...")
        state_dict = model.state_dict()

        # Create name mapping to shorten long names (GGUF has 64 char limit)
        name_mapping = {}

        for name, tensor in state_dict.items():
            # Shorten tensor name to fit GGUF's 64 character limit
            short_name = name

            # Apply common abbreviations for T5
            short_name = short_name.replace("encoder.block.", "enc.b.")
            short_name = short_name.replace("decoder.block.", "dec.b.")
            short_name = short_name.replace("layer_norm", "ln")
            short_name = short_name.replace("attention", "attn")
            short_name = short_name.replace("SelfAttention", "self_attn")
            short_name = short_name.replace("EncDecAttention", "cross_attn")
            short_name = short_name.replace("DenseReluDense", "ffn")
            short_name = short_name.replace(".weight", ".w")
            short_name = short_name.replace(".bias", ".b")
            short_name = short_name.replace("relative_attention_bias", "rel_attn_bias")

            # If still too long, truncate but keep uniqueness
            if len(short_name) > 63:
                # Keep start and end, hash the middle
                hash_val = hash(name) % 10000
                short_name = f"{short_name[:28]}_{hash_val}_{short_name[-28:]}"

            # Ensure uniqueness
            if short_name in name_mapping.values():
                counter = 1
                base_name = short_name[:60]
                while f"{base_name}_{counter}" in name_mapping.values():
                    counter += 1
                short_name = f"{base_name}_{counter}"

            name_mapping[name] = short_name

            # Convert to numpy and ensure float32
            data = tensor.cpu().numpy().astype(np.float32)
            writer.add_tensor(short_name, data)

            if len(name) != len(short_name):
                print(f"  Added tensor: {name[:40]}... -> {short_name} {data.shape}")
            else:
                print(f"  Added tensor: {short_name} {data.shape}")

        # Save name mapping for reference
        mapping_file = out_path / "tensor_name_mapping.txt"
        with open(mapping_file, 'w') as f:
            f.write("# Tensor name mapping (original -> shortened)\n")
            for orig, short in sorted(name_mapping.items()):
                f.write(f"{orig} -> {short}\n")
        print(f"\nSaved tensor name mapping to: {mapping_file}")

        # Write to file
        print("Writing GGUF file...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print("✓ Conversion successful!")
        return base_gguf

    except Exception as e:
        print(f"\nError: Conversion failed: {e}")
        print("\nNote: T5 support in GGUF may be limited.")
        print("Alternative approaches:")
        print("  1. Use ONNX conversion (recommended): python quantize_to_onnx.py")
        print("  2. Use ctranslate2 for T5 optimization")
        print("  3. Use torch.compile() for PyTorch 2.0+ inference")
        import traceback
        traceback.print_exc()
        return None


def quantize_gguf(base_gguf, quant_type, out_dir):
    """Quantize GGUF file to specified quantization type."""
    if quant_type in ['F32', 'F16']:
        print(f"\n✓ No quantization needed for {quant_type}")
        return base_gguf

    out_path = Path(out_dir)
    quantized_file = out_path / f"model-{quant_type.lower()}.gguf"

    # Find quantize binary (system installation)
    quantize_bin = find_quantize_binary()

    if not quantize_bin:
        print("Error: llama-quantize binary not found!")
        print("\nInstall with:")
        print("  sudo zypper install llamacpp  # openSUSE/SUSE")
        print("  sudo dnf install llama.cpp    # Fedora/RHEL")
        print("  sudo apt install llama-cpp    # Ubuntu/Debian")
        return None

    print(f"\nQuantizing to {quant_type}...")
    print(f"Using quantize binary: {quantize_bin}")
    print(f"Output: {quantized_file}")

    cmd = [
        str(quantize_bin),
        str(base_gguf),
        str(quantized_file),
        quant_type
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nError: Quantization to {quant_type} failed!")
        return None

    print(f"✓ Quantization successful!")

    # Print file sizes
    base_size = os.path.getsize(base_gguf) / (1024 ** 2)
    quant_size = os.path.getsize(quantized_file) / (1024 ** 2)
    compression = (1 - quant_size / base_size) * 100

    print(f"\nFile sizes:")
    print(f"  Original (F32): {base_size:.1f} MB")
    print(f"  Quantized ({quant_type}): {quant_size:.1f} MB")
    print(f"  Compression: {compression:.1f}%")

    return quantized_file


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
    out_dir = args.out_dir or str(model_dir / "gguf")

    print("=" * 60)
    print("T5 Model GGUF Quantization")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Quantization type: {args.quant_type}")
    print(f"Output directory: {out_dir}")
    print("\n" + "!" * 60)
    print("WARNING: T5 quantization is NOT supported in llama.cpp!")
    print("While GGUF files can be created, llama-quantize cannot")
    print("actually quantize T5 tensors (they remain F32).")
    print()
    print("For actual quantization, use ONNX instead:")
    print("  python quantize_to_onnx.py")
    print("!" * 60)
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check for quantize binary
    quantize_bin = find_quantize_binary()
    if quantize_bin:
        print(f"✓ Found system llama-quantize: {quantize_bin}")
    else:
        print("Warning: llama-quantize not found")
        print("Install with: sudo zypper install llamacpp")

    # Convert to GGUF
    base_gguf = Path(out_dir) / "model-f32.gguf"
    if args.skip_convert and base_gguf.exists():
        print(f"\n✓ Using existing GGUF file: {base_gguf}")
    else:
        base_gguf = convert_to_gguf(model_dir, out_dir)
        if base_gguf is None:
            sys.exit(1)

    # Quantize
    quantized_file = quantize_gguf(base_gguf, args.quant_type, out_dir)
    if quantized_file is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Quantization complete!")
    print("=" * 60)
    print(f"Quantized model: {quantized_file}")
    print("\nYou can use this model with llama.cpp or compatible inference engines.")


if __name__ == "__main__":
    main()
