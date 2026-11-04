import json


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


records = load_jsonl("synthetic_training_data.jsonl")
