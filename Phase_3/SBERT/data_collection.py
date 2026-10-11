import json
from collections import defaultdict

def load_rows(path: str) -> list[tuple[int, dict]]:
    with open(path, 'r', encoding='utf8') as f:
        entries = []
        parsing_failure_count = 0

        for line in f:
            if line == '':
                continue

            try:
                parsed_row = json.loads(line.strip())
                ground_truth = json.loads(parsed_row['ground_truth'].strip())
            except json.JSONDecodeError as e:
                parsing_failure_count += 1
                continue

            index = parsed_row['example_index']
            entries.append((index, ground_truth))

    print(f"Parsing failure count: {parsing_failure_count}")
    return entries


def assert_same_ground_truth(rows_a: list, rows_b: list) -> None:
    if len(rows_a) == len(rows_b):
        for row_a, row_b in zip(rows_a, rows_b):
            assert row_a[0] == row_b[0], f'the index from list 1 is {row_a[0]} and the index from the list 2 is {row_b[0]}'
            assert row_a[1] == row_b[1], f'The ground truths at the index {row_a[0]} are not matching'
            

def group_by_error_code(rows: list) -> dict[str, list[tuple[int, str]]]:
    error_code_map = defaultdict(list)
    for index, ground_truth in rows:

        error_code = ground_truth['error_code']
        rca = ground_truth['root_cause_analysis']

        if error_code not in ['N/A', 'None', 'misc']:     #some entries dont have a proper error code and end up having these codes which are no way related
            error_code_map[error_code].append((index, rca))
        
    error_code_map = dict(sorted(error_code_map.items(), key = lambda item: len(item[1]), reverse = True))
    return error_code_map

def print_groups(groups: dict, min_size: int = 2) -> None:
    for key, value in groups.items():
        if len(value) >= min_size:
            print(f'{key}: {len(value)}')
            for i in value:
                print(i)


ft20_entries = load_rows("Phase_2/LoRA/logs/ft20_model_outputs.jsonl")
ft500_entries = load_rows("Phase_2/LoRA/logs/ft500_model_outputs.jsonl")
assert_same_ground_truth(ft20_entries, ft500_entries)
error_code_map = group_by_error_code(ft20_entries)
print_groups(error_code_map)
