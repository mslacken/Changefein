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

# Configuration
MODEL_ID = 'google-t5/t5-small'
BATCH_SIZE = 4
NUM_PROCS = 4
EPOCHS = 3
OUT_DIR = './results_t5_finetune'
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_data(examples):
    """
    Tokenize the formatted inputs and the target 'changes_diff'.
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
 
    # Tokenize targets (changes_diff)
    labels = tokenizer(
        text_target=examples['changes_diff'],
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

def main():
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
    print(f"Loading model {MODEL_ID}...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUT_DIR, 'logs'),
        logging_steps=50,
        eval_strategy='steps',
        save_steps=1000,
        eval_steps=1000,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to='none',
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(), # Use FP16 if GPU is available
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
    print(f"Saving fine-tuned model to {OUT_DIR}...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
