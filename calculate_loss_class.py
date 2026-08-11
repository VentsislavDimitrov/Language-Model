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
            1. Store model and batch tensors as instance variables.
            2. Detect available device (CUDA GPU or CPU).
            3. Validate and cap number_of_batches_in_dataloader to actual dataloader size.

        Why:
            - Device detection enables automatic GPU acceleration when available.
            - Using min() prevents attempting to process more batches than exist,
              avoiding index out-of-bounds errors.

        Args:
            gpt_model (GPTModel): The trained GPT model to evaluate.
            input_batch (Tensor): Input tensor for single batch processing.
            target_batch (Tensor): Target tensor for single batch processing.
            data_loader (Optional[DataLoader]): Dataloader for iterating over multiple batches.
                Defaults to None.
            number_of_batches_in_dataloader (int|None): Maximum number of batches to process.
                If None, uses entire dataloader length. If provided, caps to actual dataloader size.
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
            1. Move input and target batches to the appropriate device (GPU/CPU).
            2. Forward pass through the model without gradient computation.
            3. Flatten logits and targets, then compute cross-entropy loss.
            4. Return the computed loss tensor.

        Why:
            - Device allocation is necessary for efficient computation on available hardware.
            - no_grad() context manager reduces memory usage since we only need predictions.
            - Flattening combines batch and sequence dimensions for loss calculation.

        Returns:
            Tensor: Cross-entropy loss value for the batch.
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
        Calculate average loss across multiple batches from the dataloader.

        Algorithm:
            1. Initialize total_loss counter to 0.
            2. Iterate through dataloader with enumeration to track batch index.
            3. For each batch index < number_of_batches_in_dataloader:
               a. Unpack input and target tensors from the dataloader.
               b. Call calculate_loss_of_batches() to get batch loss.
               c. Accumulate loss into total_loss counter.
            4. Break iteration once limit is reached.
            5. Return NaN if dataloader is empty, else return average loss
               (total_loss / number_of_batches_in_dataloader).

        Why:
            - Enumeration provides both index and data, enabling batch limit enforcement.
            - Tuple unpacking extracts input/target pairs in a single operation.
            - Early termination via break prevents unnecessary computations.
            - NaN return for empty dataloader avoids division-by-zero errors.
            - Averaging normalizes loss across batch count for fair comparison.

        Returns:
            int|float: Average loss across processed batches, or NaN if dataloader is empty.
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








