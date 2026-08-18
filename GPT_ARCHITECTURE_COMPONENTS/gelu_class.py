from torch import pi, Tensor, tanh, tensor
from torch.nn import Module

class GELU(Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth, non-linear activation function commonly used in modern transformer
    architectures (e.g., BERT, GPT). Unlike ReLU which hard-clips negative values to zero,
    GELU applies a probabilistic gating mechanism that allows small negative values to
    contribute to learning, resulting in smoother gradients during backpropagation.

    Why use GELU over ReLU:
    - ReLU: Binary decision - "This feature is either important (positive) or useless (zero)"
    - GELU: Probabilistic - "This feature might be somewhat important (small positive) or
      slightly negative (but still useful)"

    This softer thresholding helps the model learn more nuanced patterns and often leads
    to better performance in language models and other deep architectures.

    Mathematical formulation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Example:
        >>> gelu = GELU()
        >>> x = Tensor([-1.0, 0.0, 1.0, 2.0])
        >>> output = gelu(x)
    """

    def __init__(self):
        super().__init__()


    def forward(self, input: Tensor) -> Tensor:
        """
        Compute the GELU activation using the tanh approximation formula.

        This method implements the efficient tanh-based approximation of GELU rather than
        the exact formulation involving the error function (erf). The approximation is
        computationally faster while maintaining high accuracy.

        Formula:
            GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

        Where:
            - √(2/π) ≈ 0.7978845608 (computed from torch.pi)
            - tanh: Hyperbolic tangent function, outputs values in range (-1, 1)
            - 0.044715: Empirically derived constant for the approximation

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Activated tensor with the same shape as input.

        Note:
            The name "average_of_gelu" reflects that GELU can be interpreted as the
            expected value of stochastically applying the identity or zero function.
        """


        return 0.5 * input * (1 + tanh(tensor((2/3.141592653589793) ** 0.5, device=input.device) * (input + 0.044715 * (input ** 3))))
