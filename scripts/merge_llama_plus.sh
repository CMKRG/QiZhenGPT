python scripts/merge_llama_with_chinese_lora.py \
    --base_model Chinese-LLaMA-Plus-path \
    --lora_model lora/checkpoint-3500 \
    --output_type huggingface \
    --output_dir qizhen_model/