from torch.nn import Parameter
from torch import Tensor, tensor
import numpy

def assign_parameters(manual_parameters, open_ai_parameters) -> Parameter:
    """
    Assign OpenAI pretrained weights to manual model parameters with shape validation.

    Algorithm:
    ==========
    1. Compare shapes of manual parameters and OpenAI parameters
    2. If shapes don't match: Raise ValueError with detailed mismatch information
    3. If shapes match: Convert OpenAI parameters to PyTorch tensor with same dtype as manual parameters
    4. Return as PyTorch Parameter object

    Why This Design:
    ================
    - Shape validation prevents silent errors from mismatched model architectures
    - Converting to the same dtype ensures compatibility with manual model's tensor operations
    - Returns Parameter object (not raw tensor) so it can be assigned to model weights directly
    - Useful for both reasoning and classification modes since both use the same pretrained weights
    - Ensures weight transfer is safe before proceeding with training or fine-tuning
    - T compatibility ensures correct weight orientation for linear layers

    Args:
        manual_parameters: PyTorch Parameter or Tensor from the manual model
        open_ai_parameters: NumPy array containing pretrained weights from OpenAI

    Returns:
        Parameter: PyTorch Parameter object ready for assignment to model

    Raises:
        ValueError: If manual_parameters.shape != open_ai_parameters.shape
    """
    if manual_parameters.shape !=  open_ai_parameters.shape:
        raise ValueError(f"OpenAI parameters are not equal with manual parameters!\n"
                         f"OpenAI parameters shape: {open_ai_parameters.shape}, Manual parameters shape: {manual_parameters.shape}")
    else: return Parameter(tensor(open_ai_parameters).to(manual_parameters.dtype))


class LoadWeights:
    def __init__(self, gpt_model, open_ai_parameters):
        """
        Initialize the LoadWeights utility.

        Algorithm:
        ==========
        1. Store reference to the custom GPT model
        2. Store reference to OpenAI parameter dictionary

        Why This Design:
        ================
        - Storing references allows the load_weights() method to access both sources
        - Model reference is used to assign weights to specific layers
        - Parameter dictionary contains all weights needed for transfer
        - This class is called before any training begins (reasoning or classification mode)

        Args:
            gpt_model: Custom GPTModel instance (can be in reasoning or classification mode)
            open_ai_parameters: Dictionary containing OpenAI GPT-2 weights with structure:
                               {
                                   "wpe": position embeddings,
                                   "wte": token embeddings,
                                   "blocks": [
                                       {"attn": {...}, "mlp": {...}, "ln_1": {...}, "ln_2": {...}},
                                       ...
                                   ],
                                   "g": final LN scale,
                                   "b": final LN shift
                               }
        """
        self.gpt_model = gpt_model
        self.open_ai_parameters = open_ai_parameters





    def load_weights(self):
        """
        Load all pretrained OpenAI weights into the custom GPT model.
        This method transfers weights for: embeddings, all transformer blocks,
        layer normalizations, and output head.

        Algorithm:
        ==========
        1. Load position embeddings (wpe) - maps positions to embedding vectors
        2. Load token embeddings (wte) - maps token IDs to embedding vectors
        3. For each transformer block (0 to num_layers-1):
           a. Extract combined QKV weights from OpenAI's "c_attn"
           b. Split weights into query, key, value components (3-way split along last axis)
           c. Assign query, key, value weights (transposed) and biases
           d. Assign attention output projection weights (transposed) and biases
           e. Assign feed-forward first layer weights (transposed) and biases
           f. Assign feed-forward second layer weights (transposed) and biases
           g. Assign layer normalization 1 scale and shift (ln_1)
           h. Assign layer normalization 2 scale and shift (ln_2)
        4. Load final layer normalization scale and shift
        5. Load output head weights from token embeddings (wte) for weight tying

        Why This Design:
        ================
        - Position embeddings (wpe) are loaded first to establish positional context
        - Token embeddings (wte) are loaded next to map vocabulary to embeddings
        - Combined QKV weights are split because manual model uses separate layers
        - Transposition (.T) matches manual model's linear layer weight orientation
        - OpenAI stores QKV as combined tensor, manual model expects separate tensors
        - Weight tying: Output head shares weights with token embeddings (wte) for
          parameter efficiency, a common GPT design pattern
        - After loading, model can be used for:
          * Reasoning: Full language modeling and text generation
          * Classification: Fine-tuning with frozen features and trainable last layers
        - This setup is essential for both training modes:
          * No need to train from scratch in either mode
          * Classification fine-tuning benefits from pretrained linguistic knowledge
          * Reasoning mode benefits from pretrained language understanding

        Raises:
            ValueError: If any parameter shapes don't match (via assign_parameters())
            IndexError: If number of blocks mismatch between model and OpenAI parameters
            KeyError: If expected keys are missing from open_ai_parameters dictionary

        Note:
            This method modifies the model in-place. After loading, the model is ready
            for either training mode. For classification_mode=True, freezing parameters
            should be done after calling this method (via freeze_parameters_for_fine_tuning()).
        """

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