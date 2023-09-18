import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters

def main():
    """ Finetune a llama model on a dataset."""
    output_dir = "./results"
    model_name = "codellama/CodeLlama-34b-hf"
    dataset = Dataset.from_file("data/arxiv_summary_prompts/data-00000-of-00001.arrow")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=find_all_linear_names(base_model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(base_model)

    # Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=300,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        warmup_ratio=0.05,
    )

    trainer = SFTTrainer(
        base_model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=512,
        args=training_args
    )

    trainer.train()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}. Done!")

if __name__ == "__main__":
    main()