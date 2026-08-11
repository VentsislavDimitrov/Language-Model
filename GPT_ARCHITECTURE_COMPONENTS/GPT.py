
from torch.nn import Module, Embedding, Dropout, Sequential, Linear
from torch import Tensor, arange, no_grad

from GPT_ARCHITECTURE_COMPONENTS.layer_normalization import LayerNormalization
from GPT_ARCHITECTURE_COMPONENTS.transformer_class import TransformerBlock


class GPTModel(Module):

    def __init__(self, configuration: dict[str, int|float|bool], classification_mode: bool = False):
        """
        Initialize the GPT Model with configuration parameters.

        Algorithm:
        ==========
        1. Store configuration dictionary for later reference
        2. Create token embedding layer to map token IDs to embedding vectors
        3. Create position embedding layer to encode sequential positions
        4. Create dropout layer for regularization
        5. Stack multiple transformer blocks sequentially
        6. Create final layer normalization for output stabilization
        7. Create output linear head to project embeddings to vocabulary logits

        Why This Design:
        ================
        - Configuration dictionary allows flexible model sizing without code changes
        - Token embeddings transform discrete tokens into continuous space where
          semantic relationships can be learned
        - Position embeddings provide sequence order information (transformers are
          position-agnostic by nature)
        - Dropout prevents overfitting by randomly dropping units during training
        - Multiple transformer blocks enable hierarchical feature learning
        - Final layer norm stabilizes the output before projection
        - Linear output head maps learned representations back to vocabulary for
          next-token prediction

        Args:
            configuration: Dictionary containing model hyperparameters including
                          vocab_size, embedding_dimension, context_length, drop_rate,
                          and number_of_layers
        """

        super().__init__()
        self.classification_mode: bool = classification_mode
        self.configuration: dict[str, int|float|bool] = configuration

        self.token_embedding: Embedding = Embedding(self.configuration["vocabulary_size"],
                                                    self.configuration["embedding_dimension"])      # Creating token embedding layer that can handle X tokens and X dimensions/
        self.position_embedding: Embedding = Embedding(self.configuration["context_length"],
                                                       self.configuration["embedding_dimension"])      # Creating position embedding layer that can handle X positions and X dimensions
        self.dropout: Dropout = Dropout(configuration["dropout_rate"])                     # Dropout layer drops out randomly selected elements from the input tensor, so to prevent overfitting

        self.transformer_blocks: Sequential = Sequential(*[TransformerBlock(self.configuration)
                            for _ in range(self.configuration["number_of_layers"])])       # Calling the transformer block class 12 times
        self.final_layer_normalization: LayerNormalization = LayerNormalization(self.configuration["embedding_dimension"])
        self.out_head: Linear = Linear(self.configuration["embedding_dimension"], 2 if classification_mode == True else self.configuration["vocabulary_size"])

    def architecture(self, input_tensor: Tensor):
        """
        Forward pass through the model architecture (internal processing).

        Algorithm:
        ==========
        1. Extract batch size and sequence length from input tensor shape
        2. Convert token IDs to embedding vectors via token embedding layer
        3. Generate position indices for the sequence length
        4. Convert position indices to position embeddings
        5. Add position embeddings to token embeddings (combining both information)
          positional context
        - Dropout applied post-embedding reduces co-adaptation and prevents overfitting
        6. Apply dropout to regularize embeddings
        7. Pass through all stacked transformer blocks for multi-head attention and FFN processing
        8. Apply final layer normalization to stabilize outputs
        9. Project embeddings to vocabulary size using output head

        Why This Design:
        ================
        - Token embedding creates meaningful vector representations from discrete IDs
        - Position indices are generated dynamically based on actual sequence length
        - Position embeddings encode positional information crucial for transformer models
        - Combining token and position embeddings gives the model both semantic and
        - Transformer blocks apply multi-head self-attention to learn token relationships
          and feed-forward networks for non-linear transformations
        - Final layer normalization stabilizes activation distributions before the output projection
        - Output head converts learned embeddings to vocabulary logits for next-token prediction

        Args:
            input_tensor: Token IDs of shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size, sequence_length, vocab_size) containing logits
            for each token position in the vocabulary
        """
        if self.classification_mode:
            self.freeze_parameters_for_fine_tuning()

        amount_of_batches, sequence_length = input_tensor.shape
        input_flow = self.token_embedding(input_tensor)                                    # Converting token_ids into 768 dimensional vector

        add_indices = arange(sequence_length, device=input_tensor.device)
        position_embedding = self.position_embedding(add_indices)                          # Placing position/indices onto each token
        input_flow = input_flow + position_embedding
        input_flow = self.dropout(input_flow)                                              # Preventing overfitting of each token's embedding, by dropping them out
        input_flow = self.transformer_blocks(input_flow)                                   # The whole input flow is going into the transformer block



        input_flow = self.final_layer_normalization(input_flow)                            # Stabilizing the output logits for each token
        return self.out_head(input_flow)                                                   # Converting the learned embeddings into logits

    def freeze_parameters_for_fine_tuning(self):
        """
        Freeze most model parameters for parameter-efficient fine-tuning.
        Only the last transformer block and final layer normalization remain trainable.
        Call this method ONCE after loading pretrained weights.
        """
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only the last transformer block and final layer norm
        for param in self.transformer_blocks[-1].parameters():
            param.requires_grad = True
        for param in self.final_layer_normalization.parameters():
            param.requires_grad = True

    def forward(self, input_tensor: Tensor):
        """
        PyTorch forward method that defines the computation for one forward pass.

        Algorithm:
        ==========
        1. Call the architecture method with the input tensor
        2. Return the output logits

        Why This Design:
        ================
        - PyTorch requires a forward() method for modules to be callable using the
          model(input) syntax and to integrate with PyTorch's autograd system
        - Delegating to architecture() method separates the interface (forward) from
          implementation (architecture) for better code organization and clarity
        - This pattern allows reusing the architecture logic while maintaining
          PyTorch conventions

        Args:
            input_tensor: Token IDs tensor of shape (batch_size, sequence_length)

        Returns:
            Output logits of shape (batch_size, sequence_length, vocab_size)
        """

        return self.architecture(input_tensor)
