from torch import Tensor, randn
from torch.nn import Sequential, Linear, Module

from GPT_ARCHITECTURE_COMPONENTS.gelu_class import GELU

class FeedForward(Module):
    """
    Feed-Forward Neural Network component used in Transformer blocks.

    The FeedForward network is a position-wise fully connected feed-forward network
    that applies two linear transformations with a GELU (Gaussian Error Linear Unit)
    activation function in between. This is a key component of transformer architectures
    and is applied independently to each position in the sequence.

    Architecture:
        Input (embedding_dimension)
            → Linear Layer 1 (expands to 4x)
            → GELU Activation
            → Linear Layer 2 (projects back to original dimension)
            → Output (embedding_dimension)

    Why this design:
    - The expansion to 4x hidden dimension increases model capacity and allows the
      network to learn more complex non-linear patterns
    - The contraction back to the original dimension maintains dimensional consistency
      for residual connections in transformer blocks
    - GELU activation provides a smooth, probabilistic gating that's more effective
      than ReLU for language models, allowing small negative values to contribute
      to learning while maintaining strong gradient flow during backpropagation
    - This pattern (expand-activate-contract) is repeated in every transformer layer,
      contributing significantly to the model's expressiveness

    Example:
        >>> config = {"embedding_dimension": 768}
        >>> ff = FeedForward(config)
        >>> x = randn(32, 10, 768)  # [batch_size, seq_len, embedding_dim]
        >>> output = ff(x)          # [32, 10, 768]

    Args:
        configuration: Dictionary containing model configuration with key:
            - "embedding_dimension": Size of input/output embeddings (e.g., 768 for GPT-2)
    """

    def __init__(self, configuration: dict[str, int|bool]):
        """
        Initialize the Feed-Forward network with two linear layers and GELU activation.

        Algorithm:
        1. Initialize parent Module class to enable PyTorch tracking
        2. Create first linear layer that expands embedding_dimension to 4x
           (projects from d to 4d)
        3. Create second linear layer that contracts back to original dimension
           (projects from 4d back to d)
        4. Combine all layers into a Sequential container for efficient forward pass

        Args:
            configuration: Dictionary with "embedding_dimension" key
        """

        super().__init__()

        # Linear layer that takes input of size, based onto the embedding dimension,
        # and outputs a vector of size 4 * embedding dimension, without activation function.
        self.linear_layer_1: Linear = Linear(configuration["embedding_dimension"],
                                   4 * configuration["embedding_dimension"])


        # Linear layer that takes input of size 4 * embedding dimension,
        # and outputs a vector of size embedding dimension, without activation function.
        self.linear_layer_2: Linear = Linear(4 * configuration["embedding_dimension"],
                                   configuration["embedding_dimension"])


        #Sequential container combining expansion, activation, and contraction
        self.all_layers_and_activation_in_one_sequential: Sequential = (
            Sequential(self.linear_layer_1, GELU(), self.linear_layer_2))

    def forward(self, input: Tensor) -> Tensor:
        return self.all_layers_and_activation_in_one_sequential(input)
