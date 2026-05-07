#!/usr/bin/env python3
"""
Simple CLI for T5 inference using CTranslate2.

This is similar to llama-cli but for T5 models.
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run T5 model inference (CTranslate2 backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s --model ./results_t5-small_finetune/ctranslate2

  # Direct prompt
  %(prog)s -m ./results_t5-small_finetune/ctranslate2 -p "generate changelog: <diff>"

  # From file
  %(prog)s -m ./results_t5-small_finetune/ctranslate2 -f input.txt

  # With custom generation settings
  %(prog)s -m ./results_t5-small_finetune/ctranslate2 -p "prompt" --max-length 256 --beam-size 4
        """
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to CTranslate2 model directory"
    )
    parser.add_argument(
        "-t", "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (default: parent directory of model)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Input prompt"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Read prompt from file"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum output length (default: 512)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for beam search (default: 1 = greedy)"
    )
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=1,
        help="Number of hypotheses to return (default: 1)"
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty for beam search (default: 1.0)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode (keep prompting)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (show timing, tokens, etc.)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    return parser.parse_args()


def load_model(model_path, tokenizer_path, device, verbose):
    """Load CTranslate2 model and tokenizer."""
    try:
        import ctranslate2
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("\nInstall with:")
        print("  pip install ctranslate2 transformers")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Determine tokenizer path
    if tokenizer_path:
        tok_path = tokenizer_path
    else:
        # Try parent directory (common layout)
        tok_path = model_path.parent
        if not (tok_path / "tokenizer_config.json").exists():
            tok_path = model_path

    if verbose:
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tok_path}")
        print(f"Device: {device}")

    # Load model
    translator = ctranslate2.Translator(
        str(model_path),
        device=device,
        compute_type="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

    if verbose:
        print(f"✓ Model loaded successfully")
        print()

    return translator, tokenizer


def generate(translator, tokenizer, prompt, max_length, beam_size, num_hypotheses, length_penalty, verbose):
    """Generate output from prompt."""
    start_time = time.time()

    # Tokenize input
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    if verbose:
        print(f"Input tokens: {len(input_tokens)}")

    # Translate
    results = translator.translate_batch(
        [input_tokens],
        max_batch_size=1,
        beam_size=beam_size,
        num_hypotheses=num_hypotheses,
        length_penalty=length_penalty,
        max_decoding_length=max_length
    )

    # Decode outputs
    outputs = []
    for hypothesis in results[0].hypotheses[:num_hypotheses]:
        output_tokens = hypothesis
        output_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True
        )
        outputs.append(output_text)

    end_time = time.time()
    inference_time = end_time - start_time

    if verbose:
        print(f"Output tokens: {len(results[0].hypotheses[0])}")
        print(f"Inference time: {inference_time*1000:.1f}ms")
        print(f"Speed: {len(results[0].hypotheses[0]) / inference_time:.1f} tokens/sec")
        print()

    return outputs, inference_time


def interactive_mode(translator, tokenizer, max_length, beam_size, num_hypotheses, length_penalty, verbose):
    """Run in interactive mode."""
    print("=" * 60)
    print("T5 Interactive Mode")
    print("=" * 60)
    print("Enter your prompts below. Type 'exit' or 'quit' to quit.")
    print("Type 'help' for generation options.")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input(">>> ")
            if not prompt:
                continue

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'help':
                print("\nGeneration settings:")
                print(f"  Max length: {max_length}")
                print(f"  Beam size: {beam_size}")
                print(f"  Num hypotheses: {num_hypotheses}")
                print(f"  Length penalty: {length_penalty}")
                print()
                continue

            outputs, inference_time = generate(
                translator, tokenizer, prompt,
                max_length, beam_size, num_hypotheses, length_penalty,
                verbose
            )

            for i, output in enumerate(outputs):
                if num_hypotheses > 1:
                    print(f"\n[Hypothesis {i+1}]")
                print(output)

            if not verbose:
                print(f"  ({inference_time*1000:.1f}ms)")
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
            continue
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    args = parse_args()

    # Load model
    translator, tokenizer = load_model(
        args.model,
        args.tokenizer,
        args.device,
        args.verbose
    )

    # Get prompt
    if args.file:
        with open(args.file, 'r') as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = None

    # Interactive mode
    if args.interactive or prompt is None:
        interactive_mode(
            translator, tokenizer,
            args.max_length, args.beam_size,
            args.num_hypotheses, args.length_penalty,
            args.verbose
        )
        return

    # Single prompt mode
    outputs, inference_time = generate(
        translator, tokenizer, prompt,
        args.max_length, args.beam_size,
        args.num_hypotheses, args.length_penalty,
        args.verbose
    )

    for i, output in enumerate(outputs):
        if args.num_hypotheses > 1:
            print(f"[Hypothesis {i+1}]")
        print(output)

    if args.verbose:
        print(f"\nInference time: {inference_time*1000:.1f}ms")


if __name__ == "__main__":
    main()
