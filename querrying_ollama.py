import urllib.request, json
from convert_instructions import format_input
from tqdm import tqdm


file_path = "instructions-with-response.json"
with open(file_path, "r") as file:
    test_data = json.load(file)

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = { "model": model,
            "seed": 123,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]
    }
    payload = json.dumps(data).encode("utf-8") #B
    request = urllib.request.Request(url, data=payload, method="POST") #C
    request.add_header("Content-Type", "application/json") #C
    response_data = ""
    with urllib.request.urlopen(request) as response: #D
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def generate_ollama_scores(json_file, json_key, model):
    scores = []
    for each_data in tqdm(json_file, desc="Scoring Output"):
        prompt = (f"Given the input: {format_input(each_data)},"
                  f"and the correct output {each_data["output"]},"
                  f"score the model output from scale between 0 and 10 where 10 is the best score")



