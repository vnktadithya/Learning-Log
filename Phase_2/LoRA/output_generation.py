from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json

def format_instruction(example):
    messages = example['messages']
    system_content = messages[0]['content']
    user_content = messages[1]['content']
    assistant_content = messages[2]['content']

    text = f'<s>[INST] {system_content}\n\n{user_content} [/INST] {assistant_content}</s>'
    return text


prompts = []
ground_truths = []
with open('data/eval_dataset_full.jsonl', 'r') as f:
    for line in f:
        example = json.loads(line.strip())
        formatted_data = format_instruction(example)
        data_split = formatted_data.split('[/INST]')
        test_input = data_split[0] + '[/INST]'
        actual_output = data_split[1]
        prompts.append(test_input)
        ground_truths.append(actual_output.replace('</s>','').strip())


MODEL_ID = "mistralai/Mistral-7B-v0.1"


# load the base model once with multi-LoRA serving enabled
llm = LLM(model=MODEL_ID, quantization="bitsandbytes", dtype="bfloat16", enable_lora=True, max_lora_rank=8, max_model_len=2048, gpu_memory_utilization=0.82, enforce_eager=True)
base_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024))
# base_model output log file is logs/base_model_output_logs
with open('logs/base_model_outputs.jsonl', 'w') as f:
    for index, result in enumerate(base_model_output):
        final_output = {
            'checkpoint': 'Base',
            'example_index': index+1,
            'input': prompts[index],
            'ground_truth': ground_truths[index],
            'generated_output': result.outputs[0].text
        }
        f.write(json.dumps(final_output) + "\n")


# create the adapter layer for ft_20
ft20_adapter = LoRARequest(lora_name="ft_20", lora_int_id=1, lora_path='checkpoints/rca_mistral_adapter_ft20')
ft20_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024), lora_request=ft20_adapter)
# ft20_model output log file is logs/ft20_model_output_logs
with open('logs/ft20_model_outputs.jsonl', 'w') as f:
    for index, result in enumerate(ft20_model_output):
        final_output = {
            'checkpoint': 'ft20',
            'example_index': index+1,
            'input': prompts[index],
            'ground_truth': ground_truths[index],
            'generated_output': result.outputs[0].text
        }
        f.write(json.dumps(final_output) + "\n")

# create the adapter layer for ft_500
ft500_adapter = LoRARequest(lora_name="ft_500", lora_int_id=1, lora_path='checkpoints/rca_mistral_adapter_ft500')
ft500_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024), lora_request=ft500_adapter)
# ft500_model output log file is logs/ft500_model_output_logs
with open('logs/ft500_model_outputs.jsonl', 'w') as f:
    for index, result in enumerate(ft500_model_output):
        final_output = {
            'checkpoint': 'ft500',
            'example_index': index+1,
            'input': prompts[index],
            'ground_truth': ground_truths[index],
            'generated_output': result.outputs[0].text
        }
        f.write(json.dumps(final_output) + "\n")
