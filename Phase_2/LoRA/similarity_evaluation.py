import json
from sentence_transformers import SentenceTransformer, util

_model = None

def _get_model(): #helper function to load the model with a global variable.
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_texts(text: str):
    return _get_model().encode(text)


EXPECTED_KEYS = {'severity', 'failing_service', 'error_code', 'root_cause_analysis', 'recommended_remediation'}

def is_strictly_valid(output: str)-> bool:
    try:
        obj = json.loads(output)
    except json.JSONDecodeError as e:
        return False

    return isinstance(obj, dict) and set(obj.keys()) == EXPECTED_KEYS

def compute_similarity(file: str):
    rca_scores = []
    remedy_scores = []
    total_count = 0
    validity_count = 0
    with open(file, 'r') as f:
        for line in f:
            total_count += 1
            data = json.loads(line.strip())
            generated_output = data['generated_output']
            ground_truth = data['ground_truth']

            if is_strictly_valid(generated_output):
                validity_count += 1
                model_output_json = json.loads(generated_output.strip())
                ground_truth_json = json.loads(ground_truth.strip())

                #rca similarity score
                model_rca = model_output_json['root_cause_analysis']
                model_output_rca_embedding = embed_texts(model_rca)
                ground_truth_rca = ground_truth_json['root_cause_analysis']
                ground_truth_rca_embedding = embed_texts(ground_truth_rca)
                rca_scores.append((util.cos_sim(model_output_rca_embedding, ground_truth_rca_embedding)).item())

                #remedy similarity score
                model_output_remedy = model_output_json['recommended_remediation']
                model_output_remedy_embedding = embed_texts(model_output_remedy)
                ground_truth_remedy = ground_truth_json['recommended_remediation']
                ground_truth_remedy_embedding = embed_texts(ground_truth_remedy)
                remedy_scores.append((util.cos_sim(model_output_remedy_embedding, ground_truth_remedy_embedding)).item())

            else:
                rca_scores.append(0.0)
                remedy_scores.append(0.0)


    return rca_scores, remedy_scores

base_model_rca_scores, base_model_remedy_scores = compute_similarity('logs/base_model_outputs.jsonl')
ft20_model_rca_scores, ft20_model_remedy_scores = compute_similarity('logs/ft20_model_outputs.jsonl')
ft500_model_rca_scores, ft500_model_remedy_scores = compute_similarity('logs/ft500_model_outputs.jsonl')

print(f'Base Model RCA Scores: {base_model_rca_scores}')
print(f'Base Model Remedy Scores: {base_model_remedy_scores}')
print("====================================================================================")
print(f'ft20 Model RCA Scores: {ft20_model_rca_scores}')
print(f'ft20 Model Remedy Scores: {ft20_model_remedy_scores}')
print("====================================================================================")
print(f'ft500 Model RCA Scores: {ft500_model_rca_scores}')
print(f'ft500 Model Remedy Scores: {ft500_model_remedy_scores}')
print("====================================================================================")
print(f'Base Model RCA score mean: {sum(base_model_rca_scores)/len(base_model_rca_scores)}')
print(f'Base Model Remedy score mean: {sum(base_model_remedy_scores)/len(base_model_remedy_scores)}')
print(f'ft20 Model RCA score mean: {sum(ft20_model_rca_scores)/len(ft20_model_rca_scores)}')
print(f'ft20 Model Remedy score mean: {sum(ft20_model_remedy_scores)/len(ft20_model_remedy_scores)}')
print(f'ft500 Model RCA score mean: {sum(ft500_model_rca_scores)/len(ft500_model_rca_scores)}')
print(f'ft500 Model Remedy score mean: {sum(ft500_model_remedy_scores)/len(ft500_model_remedy_scores)}')