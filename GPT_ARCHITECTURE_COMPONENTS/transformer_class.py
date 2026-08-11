from torch import Tensor, randn
from torch.nn import Module, Dropout, Embedding
from GPT_ARCHITECTURE_COMPONENTS.attention_mechanism_class import AttentionMechanism
from GPT_ARCHITECTURE_COMPONENTS.feedforward_class import FeedForward
from GPT_ARCHITECTURE_COMPONENTS.layer_normalization import LayerNormalization

class TransformerBlock(Module):
    """
        A single Transformer block implementing the multi-head attention + feedforward architecture
        with residual connections and layer normalization, as described in "Attention is All You Need".

        Why this architecture:
        - **Multi-Head Attention**: Allows the model to attend to information from different representation
          subspaces at different positions. Multiple heads enable the model to simultaneously focus on
          different parts of the input sequence and capture diverse semantic and syntactic relationships.
        - **Layer Normalization**: Applied before attention and feedforward layers (pre-norm architecture)
          to stabilize training by reducing internal covariate shift. Normalizing inputs to each module
          ensures consistent activation distributions, facilitating gradient flow and faster convergence.
        - **Residual Connections (Skip Connections)**: Enable training of very deep networks by allowing
          gradients to bypass layers and information to flow directly forward. These connections preserve
          the original input signal and prevent vanishing gradients during backpropagation.
        - **Position-wise Feedforward Network**: A two-layer fully connected network with intermediate
          expansion and ReLU activation. This provides additional non-linearity and parameter capacity
          for learning complex transformations within each position independently.
        - **Dropout**: Applied after attention and feedforward outputs to prevent overfitting and improve
          model generalization by randomly deactivating neurons during training.

        Algorithm (Pre-Normalization Transformer Block):
        1. **Self-Attention Pathway**:
           a. input₀ = input (save as shortcut)
           b. input₁ = LayerNorm(input₀)
           c. attention_out = MultiHeadAttention(input₁)
           d. attention_out = Dropout(attention_out)
           e. input₂ = input₀ + attention_out  (residual connection)

        2. **Feedforward Pathway**:
           a. input₃ = input₂ (save as shortcut)
           b. input₄ = LayerNorm(input₃)
           c. ff_out = FeedForward(input₄)
           d. ff_out = Dropout(ff_out)
           e. output = input₃ + ff_out  (residual connection)

        The pre-normalization architecture improves training stability compared to post-normalization
        (where normalization is applied after residual connections) and has become the standard in
        modern transformer models like GPT and BERT variants.

        Attributes:
            layer_normalization_1 (LayerNormalization): Normalizes input before multi-head attention.
            layer_normalization_2 (LayerNormalization): Normalizes input before feedforward network.
            attention_mechanism (AttentionMechanism): Multi-head self-attention mechanism.
            dropout (torch.nn.Dropout): Dropout layer for regularization.
            feed_forward (FeedForward): Position-wise feedforward network.

        Args:
            configuration (dict[str, int|bool]): Configuration dictionary containing:
                - embedding_dimension (int): Dimension of token embeddings and attention layers
                - context_length (int): Maximum sequence length (for causal masking in attention)
                - number_of_heads (int): Number of attention heads
                - dropout_rate (float): Dropout probability [0.0, 1.0)
                - qkv_bias (bool): Whether to include bias in query, key, value projections

        Example:
            >>> config = {
            ...     "embedding_dimension": 768,
            ...     "context_length": 1024,
            ...     "number_of_heads": 12,
            ...     "dropout_rate": 0.1,
            ...     "qkv_bias": False
            ... }
            >>> transformer_block = TransformerBlock(config)
            >>> x = randn(32, 1024, 768)  # (batch_size, seq_len, embedding_dim)
            >>> output = transformer_block(x)  # Same shape as input
        """

    def __init__(self, configuration: dict[str, int|bool]):
        """
        Initialize a Transformer block with multi-head attention, layer normalization, and feedforward.

        Args:
            configuration (dict[str, int|bool]): Model configuration containing:
                - embedding_dimension: Size of embeddings and hidden dimensions
                - context_length: Maximum sequence length
                - number_of_heads: Number of attention heads
                - dropout_rate: Dropout probability
                - qkv_bias: Whether to use bias in attention projections
        """

        super().__init__()

        self.layer_normalization_1 = LayerNormalization(configuration["embedding_dimension"])
        self.layer_normalization_2 = LayerNormalization(configuration["embedding_dimension"])

        self.attention_mechanism = AttentionMechanism(dimension_input=configuration["embedding_dimension"],
                                                      dimension_output=configuration["embedding_dimension"],
                                                      context_length=configuration["context_length"],
                                                      number_of_heads=configuration["number_of_heads"],
                                                      dropout_range=configuration["dropout_rate"],
                                                      qkv_bias=configuration["qkv_bias"])


        self.dropout: Dropout = Dropout(configuration["dropout_rate"])
        self.feed_forward = FeedForward(configuration)



    def transformer_functionality(self, input: Tensor) -> Tensor:
        """
        Implement the core Transformer block computation with two residual pathways.

        This method implements the pre-normalization architecture where layer normalization
        is applied BEFORE (not after) each sub-layer, improving training stability and
        enabling better gradient flow in deep models.

        Algorithm:

        **Attention Pathway**:
          1. shortcut = input (preserve original for residual connection)
          2. normalized = LayerNormalization(input)
          3. attention_output = MultiHeadAttention(normalized)
          4. attention_output = Dropout(attention_output)
          5. output = shortcut + attention_output (add residual connection)

        **Feedforward Pathway**:
          6. shortcut = output (preserve output from attention pathway)
          7. normalized = LayerNormalization(output)
          8. ff_output = FeedForward(normalized)  (typically: Linear→ReLU→Linear)
          9. ff_output = Dropout(ff_output)
         10. final_output = shortcut + ff_output (add residual connection)

        The residual connections ensure that if either the attention or feedforward
        modules produce near-zero activations, the original signal still propagates
        through the layer. This is crucial for training very deep networks.

        Why two separate pathways with residual connections:
        - Residuals allow gradients to skip over complex non-linear transformations
          during backpropagation, preventing vanishing gradients
        - Pre-normalization stabilizes training by ensuring consistent activation ranges
        - Two separate residuals (attention + feedforward) provide independent paths for
          information flow, improving model expressivity

        Args:
            input (torch.Tensor): Batch of token embeddings with shape
                (batch_size, sequence_length, embedding_dimension)

        Returns:
            torch.Tensor: Transformed embeddings with the same shape as input
                (batch_size, sequence_length, embedding_dimension)

        Example:
            >>> block = TransformerBlock(config)
            >>> x = randn(2, 512, 768)  # 2 sequences, 512 tokens each, 768-dim embeddings
            >>> output = block.transformer_functionality(x)  # Shape: (2, 512, 768)
        """

        shortcut: Tensor = input
        input: Tensor = self.layer_normalization_1(input)
        input: Tensor = self.attention_mechanism(input)
        input: Tensor = self.dropout(input)

        input: Tensor = shortcut + input
        shortcut: Tensor = input

        input: Tensor = self.layer_normalization_2(input)
        input: Tensor = self.feed_forward(input)
        input: Tensor = self.dropout(input)
        input: Tensor = shortcut + input
        return input

    def forward(self, input: Tensor) -> Tensor:
        return self.transformer_functionality(input)


