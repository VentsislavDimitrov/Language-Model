from typing import Optional
from torch import no_grad, Tensor, softmax, argmax, cat, multinomial, topk, where, tensor
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
class GenerateTokens:
    def __init__(self, gpt_model: GPTModel,
                 encoded_input: Tensor,
                 maximal_new_tokens: int,
                 context_length: int,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 eos_id: Optional[int] = None):
        """
        Initialize the text generator with a model and generation parameters.

        Algorithm:
        ==========
        1. Store the GPT model reference for later use in token prediction
        2. Store the initial encoded input token indices (starting context)
        3. Store the maximum number of tokens to generate
        4. Store the context length constraint for the model
        5. Store optional generation controls: temperature, top-k, and eos_id

        Why this design:
        ================
        - The model is stored as an instance variable so it persists across multiple
          calls to generate_tokens() without requiring re-instantiation
        - Input indices are stored to enable token generation starting from a given seed
        - Generation parameters (max tokens, context length) are stored to encapsulate
          all generation settings within the instance, making the class self-contained
          and reusable
        - Optional parameters (temperature, top_k, eos_id) default to None so the class
          can be used with simple greedy decoding when advanced sampling is not needed

        Args:
            gpt_model (GPTModel): Pre-trained GPT model for generating logits.
            encoded_input (torch.Tensor): Initial token IDs tensor of shape
                (batch_size, sequence_length) containing token indices from the tokenizer.
            maximal_new_tokens (int): Maximum number of new tokens to generate. The
                generation loop will stop after producing this many tokens.
            context_length (int): Maximum context length the model supports. Should
                match the model's context_length configuration parameter.
            temperature (float, optional): Scaling factor applied to logits before
                softmax. Values < 1.0 sharpen the distribution (more conservative),
                values > 1.0 flatten it (more diverse). If None or 0.0, falls back
                to greedy decoding via argmax. Defaults to None.
            top_k (int, optional): If set, restricts sampling to only the top-k
                highest logit tokens per step. All other logits are masked to -inf
                before softmax, ensuring only the k most likely tokens are considered.
                Defaults to None (no restriction).
            eos_id (int, optional): End-of-Sequence token ID. If the model generates
                this token, generation stops early before appending it to the output.
                Defaults to None (generation always runs for maximal_new_tokens steps).
        """


        self.gpt_model: GPTModel = gpt_model                                               # Model that will output the logits, to be processed for text_generation
        self.encoded_input: Tensor = encoded_input
        self.maximal_new_tokens: int = maximal_new_tokens                                  # Maximal amount of new token output
        self.context_length: int = context_length                                          # Maximal context length that can be handled
        self.temperature: Optional[float] = temperature                                    # With how mutch the logits to be scaled so to receive diverse output
        self.top_k: Optional[int] = top_k                                                  # How many from the biggest logits to be taken
        self.eos_id = eos_id


    def generate_tokens(self) -> Tensor:
        """
        Generate new tokens autoregressively using the GPT model.

        For each generation step, the current context is fed into the model to obtain
        logits for the next token. Optionally, top-k filtering and temperature scaling
        are applied before sampling the next token via multinomial sampling or greedy
        decoding. Generation stops either after maximal_new_tokens steps or when the
        end-of-sequence token is produced.

        Algorithm:
        ==========
        For each token generation step (from 1 to maximal_new_tokens):

            1. **Context Windowing**:
               - Slice encoded_input to keep only the last context_length tokens.
               - Prevents exceeding the model's maximum supported sequence length
                 (e.g., if the model supports 1024 tokens and the current context
                 is 1100 tokens, only the last 1024 are passed to the model).

            2. **Forward Pass**:
               - Disable gradient computation with no_grad() to save memory and
                 improve inference speed.
               - Pass the windowed context through the GPT model to get logits of
                 shape (batch_size, sequence_length, vocabulary_size).

            3. **Last Token Extraction**:
               - Slice logits[:, -1, :] to extract only the last time step.
               - Reduces shape from (batch_size, sequence_length, vocab_size)
                 to (batch_size, vocab_size), since only the next token matters.

            4. **Top-K Filtering** (optional):
               - If top_k is set, retrieve the top-k highest logit values per batch item.
               - The smallest of those top-k values becomes a threshold.
               - All logits below the threshold are replaced with -inf, so they
                 contribute zero probability after softmax.

            5. **Temperature Scaling + Multinomial Sampling** (optional):
               - If temperature > 0.0, divide logits by temperature to scale the
                 distribution, then apply softmax to get probabilities.
               - Sample the next token using multinomial sampling (num_samples=1),
                 which introduces controlled randomness into the output.
               - If temperature is None or 0.0, fall back to greedy decoding:
                 select the token with the highest logit via argmax.

            6. **Early Stopping**:
               - If the selected token matches eos_id, break immediately without
                 appending the token to the output.

            7. **Context Extension**:
               - Concatenate the new token to encoded_input along the sequence
                 dimension (dim=1) so it becomes part of the context for the
                 next generation step.

        Why this design:
        ================
        - **Top-K Filtering**: Limits the candidate pool to only the most probable
          tokens, preventing the model from sampling low-quality or incoherent tokens
          that have very small but non-zero probability.
        - **Temperature Scaling**: Controls the sharpness of the probability
          distribution. Low temperature yields focused, repetitive output; high
          temperature yields diverse, creative output.
        - **Multinomial Sampling vs. Greedy Decoding**: Multinomial sampling
          introduces stochasticity, making generation more varied across runs.
          Greedy decoding (argmax) is fully deterministic and faster, suitable
          when reproducibility is required.
        - **Early Stopping via eos_id**: Allows generation to terminate naturally
          when the model signals completion, rather than always generating the
          maximum number of tokens.
        - **Context Windowing**: Respects the model's fixed maximum sequence length
          and prevents memory overflow during long generation runs.
        - **Disabling Gradients**: Inference does not require backpropagation, so
          no_grad() reduces memory usage and speeds up computation.

        Returns:
            torch.Tensor: Token indices of shape
                (batch_size, original_length + tokens_generated), containing the
                original input tokens followed by all newly generated tokens. If
                eos_id is encountered early, the output will be shorter than
                original_length + maximal_new_tokens.

        Example:
            >>> generator = GenerateTokens(gpt_model, initial_tokens,
            ...                            maximal_new_tokens=10,
            ...                            context_length=1024,
            ...                            temperature=0.8,
            ...                            top_k=50,
            ...                            eos_id=50256)
            >>> output = generator.generate_tokens()
            >>> # If initial_tokens has shape (1, 4), output has shape (1, 4+n)
            >>> # where n <= 10, depending on whether eos_id was encountered
        """


        for each_new_token in range(self.maximal_new_tokens):
            context_size = self.encoded_input[:, -self.context_length:]                    # Slicing the current context if it exceeds the supported context size (E.g., if LLM supports only 5 tokens, and the context size is 10 then only the last 5 tokens are used as context)
            with no_grad(): logits: Tensor = self.gpt_model(context_size)                  # Disabling gradient calculations after calculating the logits from the GPT Model
            logits: Tensor = logits[:, -1, :]                                              # Focus only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)

            # Applying top_k algorithm of amount of values are set
            if self.top_k is not None:
                top_logits, top_logits_indices = topk(logits, k=self.top_k)                # Finding the biggest logits into the tensor, based onto the amount that was inputted in self.top_k
                last_element = top_logits[:, -1].unsqueeze(-1)                             # Taking the last element (which is the smallest) so to be used for converting the elements that are smaller than it
                logits = where(logits < last_element,
                               tensor(float("-inf")).to(logits.device),
                               logits)                                                     # Replacing the logit values that are smaller than the smallest element of top_logits from after applied top_k algorithm

            # Applying the temperature scaling algorithm if the temperature is set
            if self.temperature is not None and self.temperature > 0.0:
                logits = logits / self.temperature                                         # Scailing the logits, so to apply the temperature scailing algorithm
                probability_values = softmax(logits, dim=-1)                               # Converting all logits in tensor into probability values
                biggest_value = multinomial(probability_values, num_samples=1)             # Choosing the next probability value, so to be added in


            else: biggest_value = argmax(logits, dim=-1, keepdim=True)                     # In case temperature and amount of top_k values are not set, it will be used Greedy Decoding algorithm
            if self.eos_id is not None and biggest_value.item() == self.eos_id: break      # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            self.encoded_input = cat((self.encoded_input,biggest_value),dim=1)

        return self.encoded_input


