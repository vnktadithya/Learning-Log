from vllm import LLM

# load the base model once with multi-LoRA serving enabled
llm = LLM(model="mistralai/Mistral-7B-v0.1", quantization="bitsandbytes", dtype="bfloat16", enable_lora=True, max_lora_rank=8)
#base_model_output
#ft20_model_output
#ft500_model_output


# score them on three tiers (JSON validity + exact-match, embedding similarity, LLM-judge)