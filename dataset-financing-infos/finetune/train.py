import argparse
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from data_utils import prepare_hf_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 14B on financial data using Unsloth")
    
    # Model
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-14B-Instruct", help="Base model path")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Use 4-bit quantization")
    
    # Dataset
    parser.add_argument("--dataset_pattern", type=str, default="../dataset/*.jsonl", help="Glob pattern for datasets")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save steps")
    
    return parser.parse_args()

def train(args):
    # 1. Load Model & Tokenizer using Unsloth
    logger.info(f"Loading model {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None, # Auto-detect (Float16 or Bfloat16)
        load_in_4bit = args.load_in_4bit,
    )

    # 2. Add LoRA adapters
    logger.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Optimized GC
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Load & Prepare Dataset
    logger.info(f"Preparing dataset from {args.dataset_pattern}...")
    dataset = prepare_hf_dataset([args.dataset_pattern])
    
    # Split dataset (90% train, 10% validation)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
    
    # Unsloth/Qwen chat template formatting
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5", # Supports qwen-2.5
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    )
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    # 4. Training Arguments
    logger.info("Setting up Trainer...")
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum_steps,
        warmup_ratio = 0.05,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.logging_steps,
        optim = "adamw_8bit", # Bitsandbytes 8-bit optimizer
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = args.output_dir,
        gradient_checkpointing = True, # Explicitly enable GC in Trainer
        
        # Validation & Early Stopping
        eval_strategy = "steps",
        eval_steps = args.save_steps, # Evaluate every time we save
        save_strategy = "steps",
        save_steps = args.save_steps,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        
        logging_dir = f"{args.output_dir}/logs",
        report_to = "wandb", # or "tensorboard"
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can set to True for speedup if short sequences
        args = training_args,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 6. Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info(f"Training complete. Stats: {trainer_stats}")

    # 7. Save Model
    logger.info("Saving model...")
    model.save_pretrained_merged(f"{args.output_dir}/model_merged", tokenizer, save_method = "merged_16bit")
    model.save_pretrained_merged(f"{args.output_dir}/model_gguf", tokenizer, save_method = "gguf_q4_k_m") # Export GGUF

    # 8. Inference Test
    logger.info("Running inference test...")
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    messages = [
        {"role": "system", "content": "Você é um analista financeiro."},
        {"role": "user", "content": "Quais foram os principais eventos econômicos de 2025?"}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
    response = tokenizer.batch_decode(outputs)
    logger.info(f"Inference Output: {response[0]}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
