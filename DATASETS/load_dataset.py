import json

def load_instructions_dataset(file_path = "DATASETS/instructions.json"):

    with open(file_path, "r", encoding="utf-8") as file: read_file = file.read()
    with open(file_path, "r") as file: data = json.load(file)
    return data