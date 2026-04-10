import argparse
import torch
import os
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset
from preprocess import preprocess_function as format_inputs
from preprocess import preprocess_target_function as format_targets

# Default Configuration
DEFAULT_MODEL_ID = 'google-t5/t5-small'
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE * GRAD_ACC
NUM_PROCS = 4
EPOCHS = 50
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model on changelog data")
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_ID,
        help=f"The model ID to use from Hugging Face (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE,
        help=f"Batch size per device (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default=None,
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=MAX_INPUT_LENGTH,
        help=f"Maximum length for input sequences (default: {MAX_INPUT_LENGTH})"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=MAX_TARGET_LENGTH,
        help=f"Maximum length for target sequences (default: {MAX_TARGET_LENGTH})"
    )
    parser.add_argument(
        "--disable_fp16",
        action="store_true",
        help="Disable FP16 mixed precision training even if CUDA is available"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_id = args.model
    model_name = model_id.split('/')[-1]
    out_dir = args.out_dir or f'./results_{model_name}_finetune'
    
    # Determine if FP16 should be used
    use_fp16 = torch.cuda.is_available() and not args.disable_fp16

    print(f"Using model: {model_id}")
    print(f"Output directory: {out_dir}")
    print(f"Max input length: {args.max_input_length}")
    print(f"Max target length: {args.max_target_length}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_acc}")
    print(f"Mixed Precision (FP16): {use_fp16}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'additional_special_tokens': ['\n']})

    def preprocess_data(examples):
        inputs = format_inputs(examples, tokenizer, max_length=args.max_input_length)
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        targets = format_targets(examples)
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        return model_inputs

    # Load dataset
    dataset = load_dataset('json', data_files="changes.json")['train']
    
    # Filter out examples where the target is too long.
    # This ensures the model always learns an EOS token for every sample.
    def filter_long_targets(example):
        target = format_targets({"changes_diff": [example["changes_diff"]]})[0]
        # Rough estimate: 1 word ~= 1.3 tokens. Use a safe margin.
        return len(tokenizer.encode(target)) <= args.max_target_length

    print(f"Original dataset size: {len(dataset)}")
    dataset = dataset.filter(filter_long_targets)
    print(f"Filtered dataset size: {len(dataset)}")

    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    tokenized_train = dataset_split['train'].map(
        preprocess_data, batched=True, num_proc=NUM_PROCS, remove_columns=dataset.column_names
    )
    tokenized_valid = dataset_split['test'].map(
        preprocess_data, batched=True, num_proc=NUM_PROCS, remove_columns=dataset.column_names
    )

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        logging_steps=10,
        eval_strategy='steps',
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to='tensorboard',
        fp16=use_fp16,
        optim="adafactor",
        label_smoothing_factor=0.1, # Prevent overconfidence
        dataloader_num_workers=2
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] # More patience for large models
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
