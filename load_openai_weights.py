from torch.nn import Parameter
from torch import Tensor, tensor
import numpy

def assign_parameters(manual_parameters, open_ai_parameters) -> Parameter:
    if manual_parameters.shape !=  open_ai_parameters.shape:
        raise ValueError(f"OpenAI parameters are not equal with manual parameters!\n"
                         f"OpenAI parameters shape: {open_ai_parameters.shape}, Manual parameters shape: {manual_parameters.shape}")
    else: return Parameter(tensor(open_ai_parameters).to(manual_parameters.dtype))


class LoadWeights:
    def __init__(self, gpt_model, open_ai_parameters):
        self.gpt_model = gpt_model
        self.open_ai_parameters = open_ai_parameters





    def load_weights(self):

        self.gpt_model.position_embedding.weight = assign_parameters(self.gpt_model.position_embedding.weight, self.open_ai_parameters["wpe"])
        self.gpt_model.token_embedding.weight = assign_parameters(self.gpt_model.token_embedding.weight,self.open_ai_parameters["wte"])

        for each_block in range(len(self.open_ai_parameters["blocks"])):
            weight_query, weight_key, weight_value = numpy.split(
                (self.open_ai_parameters                                                   # Getting the whole dictionary
                ["blocks"]                                                                 # Taking the list with key "blocks"
                [each_block]                                                               # Taking each_element in list, connected with "blocks" keys
                ["attn"]                                                                   # Taking the attention data of the block we are located on
                ["c_attn"]                                                                 # Combined query, key, value data
                )["w"], 3, axis=-1)

            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_query.weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_query.weight, weight_query.T)
            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_key.weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_key.weight, weight_key.T)
            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_value.weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_value.weight, weight_value.T)

            bias_query, bias_key, bias_value = numpy.split(
                (self.open_ai_parameters                                                   # Getting the whole dictionary
                ["blocks"]                                                                 # Taking the list with key "blocks"
                [each_block]                                                               # Taking each_element in list, connected with "blocks" keys
                ["attn"]                                                                   # Taking the attention data of the block we are located on
                ["c_attn"]                                                                 # Combined query, key, value data
                )["b"], 3, axis=-1)

            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_query.bias = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_query.bias, bias_query)
            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_key.bias = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_key.bias, bias_key)
            self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_value.bias = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.w_value.bias, bias_value)

            self.gpt_model.transformer_blocks[each_block].attention_mechanism.out_projection.weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.out_projection.weight, self.open_ai_parameters["blocks"][each_block]["attn"]["c_proj"]["w"].T)
            self.gpt_model.transformer_blocks[each_block].attention_mechanism.out_projection.bias = assign_parameters(self.gpt_model.transformer_blocks[each_block].attention_mechanism.out_projection.bias,self.open_ai_parameters["blocks"][each_block]["attn"]["c_proj"]["b"])

            self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[0].weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[0].weight, self.open_ai_parameters["blocks"][each_block]["mlp"]["c_fc"]["w"].T)
            self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[0].bias =  assign_parameters(self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[0].bias, self.open_ai_parameters["blocks"][each_block]["mlp"]["c_fc"]["b"])

            self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[2].weight = assign_parameters(self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[2].weight,self.open_ai_parameters["blocks"][each_block]["mlp"]["c_proj"]["w"].T)
            self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[2].bias = assign_parameters(self.gpt_model.transformer_blocks[each_block].feed_forward.all_layers_and_activation_in_one_sequential[2].bias,self.open_ai_parameters["blocks"][each_block]["mlp"]["c_proj"]["b"])

            self.gpt_model.transformer_blocks[each_block].layer_normalization_1.scale = assign_parameters(self.gpt_model.transformer_blocks[each_block].layer_normalization_1.scale, self.open_ai_parameters["blocks"][each_block]["ln_1"]["g"])
            self.gpt_model.transformer_blocks[each_block].layer_normalization_1.shift = assign_parameters(self.gpt_model.transformer_blocks[each_block].layer_normalization_1.shift, self.open_ai_parameters["blocks"][each_block]["ln_1"]["b"])

            self.gpt_model.transformer_blocks[each_block].layer_normalization_2.scale = assign_parameters(self.gpt_model.transformer_blocks[each_block].layer_normalization_2.scale, self.open_ai_parameters["blocks"][each_block]["ln_2"]["g"])
            self.gpt_model.transformer_blocks[each_block].layer_normalization_2.shift = assign_parameters(self.gpt_model.transformer_blocks[each_block].layer_normalization_2.shift,self.open_ai_parameters["blocks"][each_block]["ln_2"]["b"])

        self.gpt_model.final_layer_normalization.scale = assign_parameters(self.gpt_model.final_layer_normalization.scale, self.open_ai_parameters["g"])
        self.gpt_model.final_layer_normalization.shift = assign_parameters(self.gpt_model.final_layer_normalization.shift, self.open_ai_parameters["b"])
        self.gpt_model.out_head.weight = assign_parameters(self.gpt_model.out_head.weight, self.open_ai_parameters["wte"])