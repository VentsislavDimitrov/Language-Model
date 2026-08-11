import json, os, urllib
from flatbuffers.packer import int64
from torch import tensor, cat, stack, nonzero, from_numpy, Tensor
from torch.utils.data import Dataset
from converter_class import Converter

def format_input(input):
    instruction_text = f"\n\n### Instruction:\n{input['instruction']}" if input['instruction'] else ""
    input_text = f"\n\n### Input:\n{input['input']}" if input['input'] else ""
    return instruction_text + input_text


class ConvertToInstructions(Dataset):
    def __init__(self, data):
        self.data = data
        self.converter = Converter()

        self.tokens: list[Tensor] = []                                                     # Storing all the encoded input from each dictionary in dataset

        for each_data in self.data:
            instruction_text = f"\n\n### Instruction:\n{each_data['instruction']}" \
                if each_data['instruction'] else ""                                        # Formatting each instruction of each_data from the dictionary/json file into Alpaca Style Template
            input_text = f"\n\n### Input:\n{each_data['input']}" \
                if each_data['input'] else ""                                              # Formatting each input of each_data from the dictionary/json file into Alpaca Style Template
            formatted_text = instruction_text + input_text
            response_text = f"\n\n### Response:\n{each_data['output']}"                    # Formatting the expected response into Alpaca Style Template
            full_text = formatted_text + response_text                                     # Combining the formatted dictionary into one string so it to be encoded by the tokenizer
            self.tokens.append(self.converter.text_to_token_ids(full_text,
                                                                add_batch_dim=False))      # Inserting the encoded Alpaca Template for each input in JSON file


    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self):
        return len(self.data)