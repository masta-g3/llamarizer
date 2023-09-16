# 🦙 Llamarizer
Custom llamarizer.

1. `python finetune.py`
2. `python merge_peft_adapters.py - device cpu - base_model_name_or_path codellama/CodeLlama-7b-hf - peft_model_path ./results/final_checkpoint - output_dir ./merged_models/`
3. Quantize.  
    a. GGUF: https://github.com/ggerganov/llama.cpp/discussions/2948  
    b. GPTQ: `python quantize_gptq.py`