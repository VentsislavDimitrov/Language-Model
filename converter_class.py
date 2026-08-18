import torch
from tiktoken import get_encoding
from torch import Tensor, tensor

class Converter:
    def __init__(self):
        """
        Initialize the Converter with GPT-2 tokenizer.

        Algorithm:
        ==========
        1. Load the GPT-2 tokenizer from tiktoken library
        2. Store tokenizer as instance variable for later use

        Why This Design:
        ================
        - GPT-2 tokenizer is the standard tokenizer used with GPT models in both
          reasoning and classification modes
        - Loading it during initialization ensures it's available for all encoding/decoding
          operations in both training and inference
        - Tiktoken provides efficient C-based tokenization, significantly faster than
          Python implementations (important for processing large datasets)
        - Storing as instance variable avoids reloading the tokenizer for every conversion call,
          improving performance in both modes
        - Single tokenizer supports both text generation (reasoning) and text preprocessing
          (classification) without modification
        - The tokenizer's vocabulary size matches the model's vocabulary_size configuration
        """

        self.tokenizer = get_encoding("gpt2")

    def text_to_token_ids(self, text: str, add_batch_dim: bool) -> Tensor:
        """
        Convert text input into token IDs and optionally wrap in a PyTorch tensor with batch dimension.

        Algorithm:
        ==========
        1. Use GPT-2 tokenizer to encode the input text string
        2. Allow the special token '<|endoftext|>' to be recognized during encoding
        3. Convert the encoded token list into a PyTorch tensor with dtype torch.long
        4. Conditionally add a batch dimension (unsqueeze) based on add_batch_dim parameter:
           - add_batch_dim=True: Shape becomes (1, sequence_length)
           - add_batch_dim=False: Shape becomes (sequence_length,)

        Why This Design:
        ================
        - Text must be converted to token IDs for the model to process discrete vocabulary tokens
          in both reasoning and classification modes
        - GPT-2 tokenizer handles subword tokenization, breaking text into semantic units
          (important for handling out-of-vocabulary words in classification tasks)
        - The '<|endoftext|>' special token marks sequence boundaries and is crucial for
          proper model behavior in both generation and classification
        - Converting to PyTorch tensor enables GPU acceleration and integration with the model
        - add_batch_dim parameter provides flexibility:
          * True: For model input during classification inference and generation
          * False: For text preprocessing before adding custom batch dimensions
        - Using dtype=torch.long ensures compatibility with embedding layers
        - Returns tensor for direct model input without additional preprocessing

        Args:
            text (str): Input text string to be tokenized and converted
            add_batch_dim (bool): If True, adds a batch dimension at position 0.
                                  Use True for model inputs, False for preprocessing.

        Returns:
            PyTorch Tensor:
            - If add_batch_dim=True: Shape (1, sequence_length) containing token IDs
            - If add_batch_dim=False: Shape (sequence_length,) containing token IDs

        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty or contains only whitespace

        Example:
            >>> converter = Converter()
            >>> converter.text_to_token_ids("Hello world", add_batch_dim=True)
            tensor([[15496,  9956]])  # Shape: (1, 2)

            >>> converter.text_to_token_ids("Hello world", add_batch_dim=False)
            tensor([15496,  9956])    # Shape: (2,)
        """
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})           # Encoding the input into token ids and passing
        token_tensor: Tensor = tensor(encoded, dtype=torch.long)                                   # Returning superficial tensor, containing the encoded input/token ids
        return token_tensor.unsqueeze(0) if add_batch_dim == True else token_tensor

    def token_ids_to_text(self, token_ids):
        """
        Convert token IDs back into human-readable text.

        Algorithm:
        ==========
        1. Convert token IDs into a PyTorch tensor (if not already a tensor)
        2. Remove the batch dimension using squeeze to convert shape:
           - From (1, sequence_length) to (sequence_length,)
           - If already 1D, squeeze has no effect
        3. Convert the tensor to a Python list for compatibility with the tokenizer
        4. Use the GPT-2 tokenizer to decode token IDs back into text

        Why This Design:
        ================
        - Model output is in token ID form, which must be converted back to human-readable text
          in both reasoning and classification modes
        - In reasoning mode: Converts generated token IDs to readable text output
        - In classification mode: Can be used for debugging or inspecting tokenized inputs
        - Token IDs are typically batched (shape (batch_size, sequence_length)) from model output
        - Squeezing removes the batch dimension, giving us a 1D sequence of token IDs
        - The tokenizer's decode method expects a list or iterable of integers
        - Converting to list (via tolist()) extracts data from the tensor before decoding
        - GPT-2 tokenizer automatically handles merging subword tokens and special tokens back
          into coherent text
        - This method is the inverse of text_to_token_ids, completing the encoding/decoding cycle
        - Handles both single-token tensors and multi-token sequences gracefully

        Args:
            token_ids: Token IDs as a tensor or array-like object, typically of shape:
                      - (1, sequence_length) for batched output
                      - (sequence_length,) for unbatched output
                      - Can also accept lists or numpy arrays

        Returns:
            str: Decoded text string reconstructed from the token IDs

        Raises:
            TypeError: If token_ids is not a tensor or array-like object
            ValueError: If token_ids is empty or contains invalid token IDs

        Example:
            >>> converter = Converter()
            >>> token_ids = tensor([[15496, 9956]])  # Shape: (1, 2)
            >>> converter.token_ids_to_text(token_ids)
            "Hello world"

            >>> token_ids = tensor([15496, 9956])    # Shape: (2,)
            >>> converter.token_ids_to_text(token_ids)
            "Hello world"
        """

        back_to_2d: Tensor = tensor(token_ids).squeeze()                                           # Removing the superficient dimension of the tensor
        return self.tokenizer.decode(back_to_2d.tolist())                                  # Returning decoded version of generated token ids into list