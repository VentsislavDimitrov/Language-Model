from torch.nn import Parameter, Module
from torch import ones, zeros, Tensor

class LayerNormalization(Module):
    """
    Layer Normalization module that normalizes activations across the feature dimension.

    Layer normalization is a critical technique in deep learning that normalizes the
    output of a neural network layer to have a mean of zero and a standard deviation
    of one. This normalization is essential for several reasons:

    1. **Stable Training**: It reduces internal covariate shift, allowing for higher
       learning rates and faster convergence during training.

    2. **Improved Gradient Flow**: By normalizing activations, layer normalization
       prevents the vanishing/exploding gradient problem, enabling deeper networks
       to train effectively.

    3. **Reduced Sensitivity to Initialization**: Normalization makes the network less
       sensitive to weight initialization, improving robustness.

    4. **Essential for Transformers**: Layer normalization is a core component of
       transformer architectures (like GPT), where it stabilizes multi-head attention
       and feed-forward computations.

    The normalization is followed by a learnable affine transformation (scale and shift)
    that allows the model to recover the original distribution if needed, enabling the
    network to adapt to different feature ranges.

    Args:
        embedding_dimension (int): The size of the feature dimension to normalize over.
                                   This determines the shape of the learnable scale and
                                   shift parameters.

    Attributes:
        epsilon (float): A small constant (1e-5) added to the variance to prevent
                        division by zero and ensure numerical stability.
        scale (torch.nn.Parameter): Learnable scale (gamma) parameter initialized to ones.
                                    Multiplies the normalized output.
        shift (torch.nn.Parameter): Learnable shift (beta) parameter initialized to zeros.
                                    Added to the scaled normalized output.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.epsilon: float = 0.00001                                                        # Small value to avoid division by zero

        #Two trainable parameters are created for the normalization.
        self.scale: Parameter = Parameter(ones(embedding_dimension))
        self.shift: Parameter = Parameter(zeros(embedding_dimension))


    def forward(self, input: Tensor) -> Tensor:
        """
        Apply layer normalization to the input tensor.

        This method computes the mean and variance across the last dimension (features),
        normalizes the input, and then applies a learnable affine transformation
        (scale and shift).

        Args:
            input (torch.Tensor): Input tensor of shape (..., embedding_dimension).
                                 The normalization is computed over the last dimension.

        Returns:
            torch.Tensor: Normalized and transformed tensor with the same shape as input.
                         Each sample has a normalized feature distribution scaled and
                         shifted by learnable parameters.
        """


        mean: Tensor = input.mean(dim=-1, keepdim=True)                                    # Finding the average logit of each row (-1 or 1 calculates the mean of each column)
        variance: Tensor = input.var(dim=-1, keepdim=True, unbiased=False)                 # Finding the variance logit of each row (-1 or 1 calculates the variance of each column)

        return self.scale * self.normalized_input(input, mean, variance) + self.shift


    

    def normalized_input(self, input: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        """
        Normalize the input tensor using computed mean and variance.

        Applies the standardization formula: (input - mean) / sqrt(variance + epsilon)
        to transform the input to have zero mean and unit variance.

        Args:
            input (torch.Tensor): Input tensor to be normalized.
            mean (torch.Tensor): Mean computed across the feature dimension.
            variance (torch.Tensor): Variance computed across the feature dimension.

        Returns:
            torch.Tensor: Normalized tensor with zero mean and unit variance across
                         the feature dimension.
        """

        return (input - mean) / ((variance + self.epsilon) ** 0.5)

    # def lr_test(self, x):
    #     """
    #     Tested LayerNorm to ensure it mathematically transforms ANY input
    #     to have mean=0 and variance=1 -
    #     which is CRITICAL for stable neural network training!
    #     """
    #     output = self.forward(x)
    #     mean = output.mean(dim=-1, keepdim=True)
    #     variance = output.var(dim=-1, keepdim=True, unbiased=False)
    #     return mean, variance


























# def demostrate_layer_normalization():
#     batch = torch.randn(5, 5)
#     layer = Sequential(Linear(5, 6), GELU())
#     output = layer(batch)
#     mean = output.mean(dim=-1, keepdim=True)
#     variance = (output.var(dim=-1, keepdim=True))
#
#     logging.info(f"Mean: {mean}\n"
#                  f"Variance: {variance}")
#
#     normalized_input = (output - mean) / (variance ** 0.5)
#     mean = normalized_input.mean(dim=-1, keepdim=True)
#     variance = normalized_input.var(dim=-1, keepdim=True)
#
#     logging.info(f"Normalized Input: {normalized_input}\n"
#                  f"Mean: {mean}\n"
#                  f"Variance: {variance}"
#                  )