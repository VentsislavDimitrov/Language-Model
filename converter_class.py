import torch
from tiktoken import get_encoding
from torch import Tensor, tensor
from typing import Optional

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
        - GPT-2 tokenizer is the standard tokenizer used with GPT models
        - Loading it during initialization ensures it's available for all encoding/decoding operations
        - Tiktoken provides efficient C-based tokenization, significantly faster than
          Python implementations
        - Storing as instance variable avoids reloading the tokenizer for every conversion call
        """

        self.tokenizer = get_encoding("gpt2")

    def text_to_token_ids(self, text: str, add_batch_dim: bool):
        """
        Convert text input into token IDs and wrap in a PyTorch tensor with batch dimension.

        Algorithm:
        ==========
        1. Use GPT-2 tokenizer to encode the input text string
        2. Allow the special token '<|endoftext|>' to be recognized during encoding
        3. Convert the encoded token list into a PyTorch tensor
        4. Add a batch dimension (unsqueeze) to make shape (1, sequence_length)

        Why This Design:
        ================
        - Text must be converted to token IDs for the model to process discrete vocabulary tokens
        - GPT-2 tokenizer handles subword tokenization, breaking text into semantic units
        - The '<|endoftext|>' special token marks sequence boundaries and is crucial for
          proper model behavior
        - Converting to PyTorch tensor enables GPU acceleration and integration with the model
        - Adding batch dimension (unsqueeze(0)) is required because models expect batched inputs
          of shape (batch_size, sequence_length) even for single samples
        - Returning as tensor allows direct input to the GPT model without additional preprocessing

        Args:
            text: Input text string to be tokenized and converted

        Returns:
            PyTorch Tensor of shape (1, sequence_length) containing token IDs from the GPT-2 vocabulary

        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty or contains only whitespace
        """
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})           # Encoding the input into token ids and passing
        token_tensor = tensor(encoded, dtype=torch.long)                                   # Returning superficial tensor, containing the encoded input/token ids
        return token_tensor.unsqueeze(0) if add_batch_dim == True else token_tensor

    def token_ids_to_text(self, token_ids):
        """
        Convert token IDs back into human-readable text.

        Algorithm:
        ==========
        1. Convert token IDs into a PyTorch tensor (if not already)
        2. Remove the batch dimension using squeeze to convert shape (1, sequence_length) to (sequence_length,)
        3. Convert the tensor to a Python list for compatibility with the tokenizer
        4. Use the GPT-2 tokenizer to decode token IDs back into text

        Why This Design:
        ================
        - Model output is in token ID form, which must be converted back to human-readable text
        - Token IDs are typically batched (shape (batch_size, sequence_length)) from model output
        - Squeezing removes the batch dimension, giving us a 1D sequence of token IDs
        - The tokenizer's decode method expects a list or iterable of integers
        - Converting to list (via tolist()) extracts data from the tensor before decoding
        - GPT-2 tokenizer automatically handles merging subword tokens and special tokens back
          into coherent text
        - This method is the inverse of text_to_token_ids, completing the encoding/decoding cycle

        Args:
            token_ids: Token IDs as a tensor or array-like object, typically of shape
                      (1, sequence_length) or (sequence_length,)

        Returns:
            Decoded text string reconstructed from the token IDs

        Raises:
            TypeError: If token_ids is not a tensor or array-like object
            ValueError: If token_ids is empty or contains invalid token IDs
        """

        back_to_2d = tensor(token_ids).squeeze()                                           # Removing the superficient dimension of the tensor
        return self.tokenizer.decode(back_to_2d.tolist())                                  # Returning decoded version of generated token ids into list