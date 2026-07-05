import json
from test import initialize_model, model_testing
from LLM_judge import evaluate_analysis

model, tokenizer = initialize_model()

with open('test_data.jsonl', 'r') as file:
    for record in file:
        record =  json.loads(record)
        input_logs = record['messages'][1]['content']
        ground_truth = record['messages'][2]['content']
        model_prediction = model_testing(record, model, tokenizer)

        analysis = evaluate_analysis(input_logs, ground_truth, model_prediction)

        print(f'Model Prediction:\n{model_prediction}')
        print(f'Judge Analysis:\n{analysis}')
        print('-----------------------------------------------------')
