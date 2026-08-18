from torch.nn.functional import cross_entropy
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from torch.utils.data import DataLoader
from torch import device, cuda, no_grad, Tensor
from typing import Optional

class CalculateLoss:
    def __init__(self, gpt_model: GPTModel,
                 device: device,
                 classification_mode: bool,
                 input_batch: Optional[Tensor] = None,
                 target_batch: Optional[Tensor] = None,
                 data_loader: Optional[DataLoader] = None,
                 number_of_batches_in_dataloader: Optional[int] = None,
                 ):
        """
        Initialize the CalculateLoss instance.

        Algorithm:
        ==========
        1. Store model reference and device information
        2. Store batch tensors (if provided for single batch processing)
        3. Store DataLoader (if provided for multi-batch processing)
        4. Store classification_mode flag to determine loss strategy
        5. Validate and store number_of_batches_in_dataloader (capped to actual size)

        Why This Design:
        ================
        - Device detection enables automatic GPU acceleration when available
        - classification_mode flag determines:
          * Loss computation: per-token vs. sequence-level
          * Output shape: vocab_size vs. 2-class
        - Supports two usage patterns:
          * Single batch: Pass input_batch and target_batch directly
          * Multi-batch: Pass data_loader for iterative processing
        - Using min() prevents attempting to process more batches than exist,
          avoiding index out-of-bounds errors
        - Optional parameters allow flexible instantiation for different use cases
        - This class is used both during training (for loss.backward()) and
          evaluation (for monitoring model performance)

        Args:
            gpt_model (GPTModel): The trained GPT model to evaluate
            device (torch.device): Device (cpu/cuda) for tensor placement
            classification_mode (bool):
                - True: Binary classification mode (2-class output head)
                - False: Language modeling/reasoning mode (vocab_size output head)
            input_batch (Optional[Tensor]): Input tensor for single batch processing.
                Required if not using data_loader. Shape: (batch_size, seq_len)
            target_batch (Optional[Tensor]): Target tensor for single batch processing.
                Required if not using data_loader. Shape varies by mode:
                - classification_mode=False: (batch_size, seq_len)
                - classification_mode=True: (batch_size,)
            data_loader (Optional[DataLoader]): DataLoader for iterating over multiple batches.
                Required if not using input_batch/target_batch
            number_of_batches_in_dataloader (int|None): Maximum number of batches to process.
                If None, uses entire DataLoader length. If provided, caps to actual size.
                Defaults to None.
        """

        self.gpt_model: GPTModel = gpt_model
        self.input_batch: Tensor = input_batch
        self.target_batch: Tensor = target_batch
        self.device = device
        self.data_loader: Optional[DataLoader] = data_loader
        self.number_of_batches_in_dataloader = number_of_batches_in_dataloader

        self.classification_mode: bool = classification_mode







    def calculate_loss_of_batches(self, input_batch = None, target_batch = None) -> Tensor:
        """
        Calculate cross-entropy loss for a single batch.

        Algorithm:
        ==========
        1. Move input and target batches to the appropriate device (GPU/CPU)
        2. Forward pass through the model to get logits
        3. Check classification_mode:
           a. If True (classification):
              - Extract logits from the last token position only
              - Compute cross-entropy between last token logits and target labels
           b. If False (reasoning/language modeling):
              - Use all token positions
              - Flatten logits and targets to (batch_size * seq_len, vocab_size)
              - Compute per-token cross-entropy loss
        4. Return the computed loss tensor

        Why This Design:
        ================
        - Device allocation is necessary for efficient computation on available hardware
        - classification_mode determines the loss computation strategy:
          * Reasoning mode: Computes loss over ALL token positions, enabling the model
            to learn to predict every next token in the sequence (language modeling)
          * Classification mode: Uses ONLY the LAST token's prediction since the model
            is fine-tuned to output the class at the final position (sentiment, topic, etc.)
        - Flattening in reasoning mode combines batch and sequence dimensions for
          efficient cross-entropy computation across all positions
        - No gradient context is not used here because this method is called during
          training (for loss.backward()) as well as evaluation
        - Returns a scalar tensor that can be backpropagated through

        Returns:
            Tensor: Cross-entropy loss value for the batch (scalar tensor)

        Raises:
            ValueError: If input_batch and target_batch are not provided and stored
                        tensors are None
            RuntimeError: If model output shape doesn't match expected dimensions
                          for the given classification_mode
        """
        if input_batch is not None and target_batch is not None:
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
        else:
            input_batch = self.input_batch.to(self.device)                                 # Allocating the input batch to be processed onto Nvidia GPU, if such is available. Otherwise cpu
            target_batch = self.target_batch.to(self.device)                               # Allocating the target batch to be processed onto Nvidia GPU, if such is available. Otherwise cpu



        if self.classification_mode == True:
            logits = self.gpt_model(input_batch)[:, -1]
            return cross_entropy(logits, target_batch)
        else:
            logits = self.gpt_model(input_batch)                                           # Inputting the input batch to the model
            logits_flat = logits.reshape(-1, logits.size(-1))                              # (batch_size * seq_len, vocab_size)
            targets_flat = target_batch.reshape(-1)                                        # (batch_size * seq_len,)  # Shape: (batch_size * seq_len,) - 1D!
            return cross_entropy(logits_flat,targets_flat)

    def calculate_loss_of_loader(self) -> int|float:
        """
        Calculate average loss across multiple batches from the DataLoader.

        Algorithm:
        ==========
        1. Initialize total_loss counter to 0
        2. If number_of_batches_in_dataloader is None, use full DataLoader length
        3. If provided, cap the value to the actual DataLoader length (min)
        4. Iterate through DataLoader with enumeration to track batch index
        5. For each batch index < number_of_batches_in_dataloader:
           a. Unpack input and target tensors from the DataLoader
           b. Call calculate_loss_of_batches() to get batch loss
           c. Accumulate loss into total_loss counter
        6. Break iteration once limit is reached
        7. Return NaN if DataLoader is empty, else return average loss
           (total_loss / number_of_batches_in_dataloader)

        Why This Design:
        ================
        - Enumeration provides both index and data, enabling batch limit enforcement
        - Tuple unpacking extracts input/target pairs in a single operation
        - Early termination via break prevents unnecessary computations
        - NaN return for empty DataLoader avoids division-by-zero errors
        - Averaging normalizes loss across batch count for fair comparison
        - This method is used during evaluation (no_grad context in caller)
        - classification_mode is passed through to calculate_loss_of_batches()
          to ensure correct loss computation strategy per mode
        - Batch limiting is useful for quick validation (evaluate_iteration)
          without processing the entire dataset

        Returns:
            int|float: Average loss across processed batches, or NaN if DataLoader is empty

        Raises:
            RuntimeError: If data_loader is None or not properly initialized
        """

        if self.number_of_batches_in_dataloader is None:
            self.number_of_batches_in_dataloader = len(self.data_loader)
        else:
            self.number_of_batches_in_dataloader = min(
                self.number_of_batches_in_dataloader, len(self.data_loader))

        total_loss: float = 0                                                                # Each loss of each batch will be added in this float counter, so it to be used for caclulating the average/loss of loader
        for each_index, (each_input_batch, each_target_batch) in enumerate(self.data_loader):# Each input batch loss of loader will be calculated by calling the "calculate_loss_of_batches"
            if each_index < self.number_of_batches_in_dataloader:
                batch_loss = self.calculate_loss_of_batches(input_batch= each_input_batch, target_batch= each_target_batch)
                total_loss += batch_loss.item()                                              # Result from lost batches is being inserted into the counter
            else: break
        return float("nan") if len(self.data_loader) == 0 \
                            else total_loss/self.number_of_batches_in_dataloader








