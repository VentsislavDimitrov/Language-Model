import json, re

import pandas, time, matplotlib.pyplot as plot
from logging import basicConfig, INFO, info

import torch
from tiktoken import get_encoding
from torch import tensor, Tensor, stack, manual_seed, log, no_grad, device, cuda, softmax, \
    argmax, multinomial, save, load
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from load_openai_weights import LoadWeights
from token_generation_class import GenerateTokens
from GPT_ARCHITECTURE_COMPONENTS.gpt_configuration import GPT2_configuration_124M_parameters_V2, GPT2_configuration_355M_parameters
from converter_class import Converter
from torch.utils.data import DataLoader
from GPT_ARCHITECTURE_COMPONENTS.gpt_download import download_and_load_gpt2
from encode_and_pad import EncodeAndPad
from classification_accuracy_class import ClassificationAccuracy
from train_model_class import TrainModel
from classify_text_class import ClassifyText
from convert_instructions import format_input, ConvertToInstructions
from DATASETS.load_dataset import load_instructions_dataset
from functools import partial
from pad_batch_function import pad_batch
from tqdm import tqdm


tokenizer = get_encoding("gpt2")

if torch.cuda.is_available():         device = device("cuda")
if torch.backends.mps.is_available(): device = device("mps")
else: device =                        device("cpu")

settings, params = download_and_load_gpt2("124M", "OPENAI_GPT2_124M_WEIGHTS")

input_text: str = str(
    "Congratulations! You won reward from your lottery ticket! Claim the reward from the link below."
)
converter = Converter()



##########################################################################################
###########################DATASET FOR INSTRUCTIONS#######################################
# gpt_model = GPTModel(GPT2_configuration_124M_parameters_V2)
# gpt_model.to(device=device)
# data = load_instructions_dataset()
#
#
# train_portion = int(len(data) * 0.85)
# test_portion = int(len(data) * 0.10)
# validation_portion = len(data) - train_portion - test_portion
#
# train_data = data[:train_portion]
# test_data = data[train_portion:train_portion + test_portion]
# validation_data = data[train_portion + test_portion:]
#
#
# train_dataset = ConvertToInstructions(train_data)
# test_dataset = ConvertToInstructions(test_data)
# validation_dataset = ConvertToInstructions(validation_data)
#
#
#
# collate_function = partial(pad_batch,
#                            device = device,
#                            allowed_max_length = GPT2_configuration_124M_parameters_V2["context_length"])
#
# load_train_data = DataLoader(dataset=train_dataset,
#                              batch_size=8,
#                              shuffle=False,
#                              drop_last=False,
#                              collate_fn=collate_function)
# load_validation_data = DataLoader(dataset=validation_dataset,
#                                   batch_size=8,
#                                   shuffle=False,
#                                   drop_last=False,
#                                   collate_fn=collate_function)
# load_test_data = DataLoader(dataset=test_dataset,
#                             batch_size=8,
#                             shuffle=False,
#                             drop_last=False,
#                             collate_fn=collate_function)
# load_weights = LoadWeights(gpt_model=gpt_model, open_ai_parameters=params)
# load_weights.load_weights()

# train_model = TrainModel(gpt_model=gpt_model,
#                          train_epoch=5,
#                          train_loader=load_train_data,
#                          validation_loader=load_validation_data,
#                          evaluate_frequency=50,
#                          evaluate_iteration=5,
#                          input_text=input_text,
#                          device=device, classification_mode = False)
##########################################################################################








##########################################################################################
##################################DATASET FOR CLASSIFICATION##############################

gpt_model = GPTModel(GPT2_configuration_124M_parameters_V2, classification_mode=True)
gpt_model.to(device=device)
train_encode_pad = EncodeAndPad(csv_file="DATASETS/train_dataset.csv",
                                tokenizer=tokenizer)
validation_encode_pad = EncodeAndPad(csv_file="DATASETS/validation_dataset.csv",
                                     tokenizer=tokenizer)
test_encode_pad = EncodeAndPad(csv_file="DATASETS/test_dataset.csv",
                               tokenizer=tokenizer)


assert train_encode_pad.max_length <= GPT2_configuration_124M_parameters_V2["context_length"], \
        (
            f"Dataset length {train_encode_pad.max_length} exceeds model's context "
            f"length {GPT2_configuration_124M_parameters_V2['context_length']}. Reinitialize data sets with "
            f"`max_length={GPT2_configuration_124M_parameters_V2['context_length']}`"
        )



load_train_data = DataLoader(dataset=train_encode_pad,
                             batch_size=8,
                             shuffle=True,
                             drop_last=True)
load_validation_data = DataLoader(dataset=validation_encode_pad,
                                  batch_size=8,
                                  shuffle=True,
                                  drop_last=True)
load_test_data = DataLoader(dataset=test_encode_pad,
                            batch_size=8,
                            shuffle=True,
                            drop_last=True)
ct = ClassifyText(text=input_text,
                  gpt_model=gpt_model,
                  device=device,
                  max_length=train_encode_pad.max_length)

train_model = TrainModel(gpt_model=gpt_model,
                         train_epoch=5,
                         train_loader=load_train_data,
                         validation_loader=load_validation_data,
                         evaluate_frequency=50,
                         evaluate_iteration=5,
                         input_text=input_text,
                         device=device, classification_mode = True)
##########################################################################################




train_accuracy = ClassificationAccuracy(data_loader=load_train_data,
                                        gpt_model=gpt_model,
                                        device=device)
validation_accuracy = ClassificationAccuracy(data_loader=load_validation_data,
                                             gpt_model=gpt_model,
                                             device=device)
test_accuracy = ClassificationAccuracy(data_loader=load_test_data,
                                       gpt_model=gpt_model,
                                       device=device)












if __name__ == "__main__":
    manual_seed(123)
    basicConfig(level=INFO, format="%(message)s")



















#####################################TRAINING PROCESS#####################################
    start_time = time.time()
    if train_model.classification_mode == True:
        train_loss_history, validation_loss_history, seen_tokens, train_accuracies, validation_accuracies = train_model.train_model()
    elif train_model.classification_mode == False:
        train_loss_history, validation_loss_history, seen_tokens = train_model.train_model()

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    info(f"Training completed in {execution_time_minutes:.2f} minutes.")
##########################################################################################

    info(ct.classify_text())

    gpt_model.eval()
    # for each_entry in test_data[:3]:
    #
    #
    #     input_text = format_input(each_entry)
    #     generated_token_ids = GenerateTokens(gpt_model=gpt_model,
    #                                          encoded_input=converter.text_to_token_ids(input_text,
    #                                          add_batch_dim=True).to(device),
    #                                          maximal_new_tokens=256,
    #                                          context_length=GPT2_configuration_124M_parameters_V2["context_length"],
    #                                          eos_id=50256,
    #                                          temperature= 1.4,
    #                                          top_k=25)
    #     generated_tokens = generated_token_ids.generate_tokens()
    #     generated_text = converter.token_ids_to_text(generated_tokens)
    #     model_response = generated_text[len(input_text):].replace("### Response:","").strip()
    #     print(input_text)
    #     print(f"\nCorrect response:\n>> {each_entry['output']}")
    #     print(f"\nModel response:\n>> {model_response.strip()}")
    #     print("-------------------------------------")

    # for i, each_entry in tqdm(enumerate(test_data), total=len(test_data)):
    #     input_text = format_input(each_entry)
    #     generated_token_ids = GenerateTokens(gpt_model=gpt_model,
    #                                          encoded_input=converter.text_to_token_ids(input_text, add_batch_dim=True).to(device),
    #                                          maximal_new_tokens=256,
    #                                          context_length=
    #                                          GPT2_configuration_124M_parameters_V2[
    #                                              "context_length"],
    #                                          eos_id=50256,
    #                                          temperature=1.4,
    #                                          top_k=25)
    #     generated_tokens = generated_token_ids.generate_tokens()
    #     generated_text = converter.token_ids_to_text(generated_tokens)
    #     model_response = generated_text[len(input_text):].replace("### Response:", "").strip()

















    # print(f"Training accuracy: {train_accuracy.calculate_accuracy() * 100:.2f}%")
    # print(f"Validation accuracy: {validation_accuracy.calculate_accuracy() * 100:.2f}%")
    # print(f"Test accuracy: {test_accuracy.calculate_accuracy() * 100:.2f}%")






    # with no_grad():
    #     print(f"Training loss: {train_loss.calculate_loss_of_loader():.2f}")
    #     print(f"Validation loss: {validation_loss.calculate_loss_of_loader():.2f}")
    #     print(f"Test loss: {test_loss.calculate_loss_of_loader():.2f}")

    # inputs = tensor([[16833, 3626, 6100],                                                  # ["every effort moves",
    #                        [40, 1107, 588]], dtype= torch.int)                                  # "I really like"]
    #
    # targets = tensor([[3626, 6100, 345],                                              # [" effort moves you",
    #                         [107, 588, 11311]], dtype = torch.int)                         # " really like chocolate"]

    # with open("dataset.txt", "r", encoding="utf-8") as file:
    #     text_data = file.read()
    #
    #
    #
    #
    #
    # range = int(len(text_data) * 0.70)

    # generative_tokens = GenerateTokens(gpt_model=gpt_model,
    #                                      encoded_input=converter_text.to(device),
    #                                      maximal_new_tokens=25,
    #                                      context_length=GPT2_configuration_124M_parameters_V2[
    #                                          "context_length"],
    #                                      temperature=0.3,
    #                                      top_k=50)
    # text_to_be_converted = generative_tokens.generate_tokens()
    # info(f"Result text: {converter.token_ids_to_text(text_to_be_converted)}")
