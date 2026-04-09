import argparse
import torch
import os
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from preprocess import preprocess_function as format_inputs
from preprocess import preprocess_target_function as format_targets

# Default Configuration
DEFAULT_MODEL_ID = 'google-t5/t5-small'
BATCH_SIZE = 4
NUM_PROCS = 4
EPOCHS = 3
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
        help=f"Batch size for training and evaluation (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default=None,
        help="Output directory for the fine-tuned model (default: ./results_t5_finetune_<model_name>)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda', 'cpu', 'cuda:0', 'mps'). If not specified, will use CUDA/ROCm if available."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_id = args.model
    model_name = model_id.split('/')[-1]
    out_dir = args.out_dir or f'./results_{model_name}_finetune'
    
    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using model: {model_id}")
    print(f"Output directory: {out_dir}")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_data(examples):
        """
        Tokenize the formatted inputs and the formatted targets.
        """
        # format_inputs returns a list of formatted strings for the model input
        inputs = format_inputs(examples)
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding='max_length'
        )
     
        # Preprocess and tokenize targets (changes_diff)
        targets = format_targets(examples)
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding='max_length'
        )
     
        # Replace padding token id with -100 so loss calculation ignores padding
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        
        return model_inputs

    # Load dataset from changes.json
    print("Loading dataset from changes.json...")
    if not os.path.exists("changes.json"):
        print("Error: changes.json not found.")
        return

    dataset = load_dataset('json', data_files="changes.json")['train']
    
    # Split dataset into train and validation (90/10 split)
    print("Splitting dataset...")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    valid_dataset = dataset_split['test']

    # Preprocess datasets
    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=NUM_PROCS,
        remove_columns=train_dataset.column_names
    )
    tokenized_valid = valid_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=NUM_PROCS,
        remove_columns=valid_dataset.column_names
    )

    # Load model
    print(f"Loading model {model_id}...")
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    # Resize model embeddings to account for the new '\n' token
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    print(f"Moving model to device: {device}")
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(out_dir, 'logs'),
        logging_steps=50,
        eval_strategy='steps',
        save_steps=1000,
        eval_steps=1000,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to='none',
        learning_rate=1e-4,
        fp16=device.startswith("cuda"), # Use FP16 if using CUDA/ROCm
        dataloader_num_workers=2
    )
 
    # Data collator for padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model and tokenizer
    print(f"Saving fine-tuned model to {out_dir}...")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done!")

if __name__ == "__main__":
    main()
