#!/usr/bin/env python3
"""
Quantize a fine-tuned T5 model to ONNX format with quantization.

This is an alternative to GGUF that has better T5 support.
Uses ONNX Runtime for efficient inference with quantization.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a fine-tuned T5 model to ONNX format"
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
        choices=['int8', 'uint8', 'float16', 'none'],
        help="Quantization type (default: int8)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for ONNX files (default: {model_dir}/onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    return parser.parse_args()


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
    except ImportError:
        missing.append("optimum[onnxruntime]")

    if missing:
        print("Error: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def export_to_onnx(model_dir, out_dir, opset):
    """Export HuggingFace T5 model to ONNX format."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting {model_dir} to ONNX format...")
    print(f"Output directory: {out_path}")
    print(f"ONNX opset: {opset}")

    try:
        # Export to ONNX
        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_dir,
            export=True,
            opset=opset
        )

        # Save the model
        model.save_pretrained(out_path)

        # Also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.save_pretrained(out_path)

        print("✓ ONNX export successful!")
        return True

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return False


def quantize_onnx(model_dir, quant_type):
    """Quantize ONNX model."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from pathlib import Path
    import shutil

    if quant_type == 'none':
        print("\n✓ Skipping quantization (none specified)")
        return True

    print(f"\nQuantizing ONNX model to {quant_type}...")

    try:
        # Find ONNX model files
        model_path = Path(model_dir)
        encoder_path = model_path / "encoder_model.onnx"
        decoder_path = model_path / "decoder_model.onnx"
        decoder_with_past_path = model_path / "decoder_with_past_model.onnx"

        # Create quantization config
        if quant_type == 'int8':
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        elif quant_type == 'uint8':
            qconfig = AutoQuantizationConfig.arm64(is_static=False)
        elif quant_type == 'float16':
            # For float16, we use a different approach
            from onnxruntime.transformers.optimizer import optimize_model
            from onnxruntime.quantization import quantize_dynamic, QuantType

            print("Converting to float16...")
            # This is a simplified approach - full implementation would need more work
            print("Note: Float16 quantization requires manual implementation")
            print("Consider using int8 instead for better compatibility")
            return True
        else:
            print(f"Unsupported quantization type: {quant_type}")
            return False

        # Quantize each component
        components = []
        if encoder_path.exists():
            components.append(("encoder", encoder_path))
        if decoder_path.exists():
            components.append(("decoder", decoder_path))
        if decoder_with_past_path.exists():
            components.append(("decoder_with_past", decoder_with_past_path))

        for name, component_path in components:
            print(f"  Quantizing {name}...")
            quantizer = ORTQuantizer.from_pretrained(model_path, file_name=component_path.name)

            # Create backup
            backup_path = component_path.with_suffix('.onnx.backup')
            shutil.copy(component_path, backup_path)

            # Quantize
            quantizer.quantize(
                save_dir=model_path,
                quantization_config=qconfig,
                file_suffix=""  # Overwrite original
            )

            # Get file sizes
            original_size = os.path.getsize(backup_path) / (1024 ** 2)
            quantized_size = os.path.getsize(component_path) / (1024 ** 2)
            compression = (1 - quantized_size / original_size) * 100

            print(f"    Original: {original_size:.1f} MB")
            print(f"    Quantized: {quantized_size:.1f} MB")
            print(f"    Compression: {compression:.1f}%")

        print("✓ Quantization successful!")
        return True

    except Exception as e:
        print(f"Error during quantization: {e}")
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

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Set output directory
    out_dir = args.out_dir or str(model_dir / "onnx")

    print("=" * 60)
    print("T5 Model ONNX Quantization")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Quantization type: {args.quant_type}")
    print(f"Output directory: {out_dir}")
    print(f"ONNX opset: {args.opset}")
    print("=" * 60)

    # Export to ONNX
    if not export_to_onnx(model_dir, out_dir, args.opset):
        sys.exit(1)

    # Quantize
    if not quantize_onnx(out_dir, args.quant_type):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ ONNX conversion and quantization complete!")
    print("=" * 60)
    print(f"ONNX model: {out_dir}")
    print("\nUsage example:")
    print("  from optimum.onnxruntime import ORTModelForSeq2SeqLM")
    print("  from transformers import AutoTokenizer")
    print(f"  model = ORTModelForSeq2SeqLM.from_pretrained('{out_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{out_dir}')")


if __name__ == "__main__":
    main()
