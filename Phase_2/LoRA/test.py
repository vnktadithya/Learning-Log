import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_formatting import format_instruction
from input_tokenizer import tokenize_function

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

# formatting test input
test_example = {"messages": [{"role": "system", "content": "You are an elite Site Reliability Engineer. Extract the root cause from the provided crash log. You must output ONLY a valid JSON object with the keys: severity, failing_service, error_code, root_cause_analysis, recommended_remediation."}, {"role": "user", "content": "user_report: Pydantic's schema generation behaves differently on Windows compared to Unix for `Path` types. Specifically, `pathlib.WindowsPath` is excluded from `GenerateSchema.match_type`, causing `PydanticSchemaGenerationError` when trying to generate a schema for a model containing a `Path` on Windows.\n\nTRACEBACK:\nPydanticJsonSchemaWarning: Default value config.toml is not JSON serializable; excluding default from JSON schema [non-serializable-default]\nPydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'pathlib.WindowsPath'>."}, {"role": "assistant", "content": "{\"severity\": \"HIGH\", \"failing_service\": \"pydantic.schema_generation\", \"error_code\": \"PydanticSchemaGenerationError\", \"root_cause_analysis\": \"Pydantic's internal schema generation logic, specifically `GenerateSchema.match_type`, did not include `pathlib.WindowsPath` in its `PATH_TYPES` list. This omission prevented `WindowsPath` instances from being correctly recognized and serialized into the JSON schema, leading to a `PydanticSchemaGenerationError` when a `BaseModel` containing a `Path` type was processed on a Windows system.\", \"recommended_remediation\": \"Add `pathlib.WindowsPath` to the `PATH_TYPES` list within Pydantic's `_internal._generate_schema.py` module to ensure consistent schema generation behavior across different operating systems for `Path` types.\"}"}]}
formatted_data = format_instruction(test_example)
test_input = formatted_data['text'].split('[/INST]')[0] + '[/INST]'
tokenized_input = tokenize_function({'text': test_input}, tokenizer)
model_input = tokenized_input.to('cuda')

# generate and decode the output
generated_ids = model.generate(**model_input)[0]
output = tokenizer.decode(generated_ids)
print(output)