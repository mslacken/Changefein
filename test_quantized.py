#!/usr/bin/env python3
"""
Test and benchmark quantized models.

This script loads a quantized model and runs inference to verify it works correctly.
"""

import argparse
import time
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test quantized T5 model"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the quantized model directory"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['onnx', 'pytorch'],
        default='onnx',
        help="Model format (default: onnx)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input text to test (default: uses sample input)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)"
    )
    return parser.parse_args()


def load_onnx_model(model_path):
    """Load ONNX quantized model."""
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: optimum or transformers not installed")
        print("Install with: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    print(f"Loading ONNX model from {model_path}...")
    model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Model loaded successfully")

    return model, tokenizer


def load_pytorch_model(model_path):
    """Load PyTorch model."""
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed")
        print("Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading PyTorch model from {model_path}...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Model loaded successfully")

    return model, tokenizer


def generate_text(model, tokenizer, input_text, max_length=512):
    """Generate text using the model."""
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=1,  # Use greedy decoding for speed
        do_sample=False
    )
    inference_time = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text, inference_time


def benchmark(model, tokenizer, input_text, num_runs=10):
    """Run performance benchmark."""
    print(f"\nRunning benchmark ({num_runs} runs)...")

    times = []
    for i in range(num_runs):
        _, inference_time = generate_text(model, tokenizer, input_text)
        times.append(inference_time)
        print(f"  Run {i+1}/{num_runs}: {inference_time*1000:.1f}ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\nBenchmark Results:")
    print(f"  Average: {avg_time*1000:.1f}ms")
    print(f"  Min: {min_time*1000:.1f}ms")
    print(f"  Max: {max_time*1000:.1f}ms")
    print(f"  Throughput: {1/avg_time:.2f} inferences/second")

    return avg_time


def get_model_size(model_path):
    """Calculate total model size."""
    import os

    total_size = 0
    model_path = Path(model_path)

    if model_path.is_file():
        return os.path.getsize(model_path)

    for file in model_path.rglob('*'):
        if file.is_file():
            total_size += os.path.getsize(file)

    return total_size


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    # Default test input
    sample_input = (
        "Generate changelog from diff:\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -10,7 +10,7 @@ def main():\n"
        "-    print('Hello World')\n"
        "+    print('Hello, World!')\n"
    )

    input_text = args.input or sample_input

    print("=" * 60)
    print("Testing Quantized T5 Model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Format: {args.format}")
    print("=" * 60)

    # Get model size
    model_size = get_model_size(model_path)
    print(f"\nModel size: {model_size / (1024**2):.1f} MB")

    # Load model
    if args.format == 'onnx':
        model, tokenizer = load_onnx_model(model_path)
    else:
        model, tokenizer = load_pytorch_model(model_path)

    # Test inference
    print("\n" + "=" * 60)
    print("Test Inference")
    print("=" * 60)
    print(f"Input ({len(input_text)} chars):")
    print(input_text[:200] + ("..." if len(input_text) > 200 else ""))
    print("\nGenerating output...")

    generated_text, inference_time = generate_text(model, tokenizer, input_text)

    print(f"\nOutput ({len(generated_text)} chars):")
    print(generated_text)
    print(f"\nInference time: {inference_time*1000:.1f}ms")

    # Benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 60)
        print("Performance Benchmark")
        print("=" * 60)
        benchmark(model, tokenizer, input_text, args.num_runs)

    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
