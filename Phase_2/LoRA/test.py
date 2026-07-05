import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_formatting import format_instruction
from input_tokenizer import tokenize_function

def initialize_model():
    bnb_config = BitsAndBytesConfig(load_in_4bit = True,                 #bits and bytes configuration to enable the properties of QLoRA: 4NF, Double Quantization, converting weights back to 16 bit floats for forward pass computations
                                    bnb_4bit_quant_type = 'nf4',
                                    bnb_4bit_use_double_quant = True,
                                    bnb_4bit_compute_dtype = torch.bfloat16
                                    )

    # model configuraiton
    model_id = "mistralai/Mistral-7B-v0.1"
    torch.cuda.empty_cache() # Clear CUDA cache to free up memory
    base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base_model, './rca_mistral_adapter')

    return model, tokenizer


def model_testing(test_example, model, tokenizer):
    # formatting test input
    formatted_data = format_instruction(test_example)
    test_input = formatted_data['text'].split('[/INST]')[0] + '[/INST]'
    tokenized_input = tokenize_function({'text': test_input}, tokenizer)
    model_input = tokenized_input.to('cuda')

    # generate and decode the output
    input_length = model_input.input_ids.shape[1]
    generated_ids = model.generate(**model_input, max_new_tokens = 300, pad_token_id=tokenizer.eos_token_id)[0]
    new_tokens = generated_ids[input_length:]
    output = tokenizer.decode(new_tokens, skip_special_tokens = True)
    return output