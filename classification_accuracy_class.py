from torch.utils.data import DataLoader
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from torch import no_grad,argmax


class ClassificationAccuracy:
    """
    A utility class for calculating classification accuracy of a GPT model on a given dataset.

    This class evaluates a GPT model's performance on a classification task by computing
    the accuracy of predictions against target labels across a specified number of batches.

    Algorithm:
    ==========
    1. Initialize with a data loader, GPT model, device, and optional batch limit
    2. Set the number of batches to evaluate (either all or a specified subset)
    3. During accuracy calculation, iterate through batches in evaluation mode
    4. For each batch, forward propagate to get logits and extract the last token's predictions
    5. Compare predicted class (argmax) against target labels
    6. Accumulate correct predictions and total examples
    7. Return the final accuracy ratio

    Why This Design:
    ================
    - Separates accuracy computation into a dedicated class for reusability and clean code
    - Supports evaluation on a subset of batches (useful for quick validation during training)
    - Uses evaluation mode (eval()) to disable dropout and other training-specific behaviors
    - Extracts only the last token's logits since classification tasks typically use the
      final position's prediction
    - Computes accuracy incrementally to handle large DATASETS without memory issues
    - Handles edge cases like empty data loaders gracefully
    - Leverages PyTorch's no_grad() context for memory-efficient inference

    Attributes:
        data_loader (DataLoader): The data loader providing input-target pairs
        gpt_model (GPTModel): The GPT model to evaluate for classification accuracy
        device (torch.device): The device (CPU/GPU) to run computations on
        number_of_batches_in_dataloader (int): Number of batches to evaluate
        correct_predictions (int): Running count of correctly classified examples
        numbers_of_examples (int): Running count of total examples processed
    """
    def __init__(self, data_loader,  gpt_model, device, number_of_batches = None):

        self.data_loader: DataLoader = data_loader
        self.gpt_model: GPTModel = gpt_model
        self.device = device


        if number_of_batches is None:
            self.number_of_batches_in_dataloader = len(data_loader)                        # Take the length from data_loader if a specific number of batches is not set
        else: self.number_of_batches_in_dataloader = min(number_of_batches,
                                                         len(data_loader))                 # Take a specific number of batches that will be used to calculate accuracy

        self.correct_predictions: int = 0
        self.numbers_of_examples: int = 0

    def calculate_accuracy(self):
        """
       Calculate the classification accuracy of the GPT model on the dataset.

       Algorithm:
       ==========
       1. Check if data loader is empty; return None if so
       2. Set model to evaluation mode (disables dropout, etc.)
       3. Iterate through batches up to the specified number
       4. Move batch data to the configured device
       5. Use no_grad() context for memory-efficient inference
       6. Forward pass to get logits for all tokens
       7. Extract logits for the last token position
       8. Apply argmax to get predicted class IDs
       9. Accumulate total examples and correct predictions
       10. Calculate and return accuracy as (correct / total)

       Why This Design:
       ================
       - Empty data loader check prevents division by zero errors
       - Evaluation mode (eval()) ensures consistent behavior vs. training mode
       - Moving data to device handles both CPU and GPU execution seamlessly
       - no_grad() context is essential for memory efficiency during inference
       - Using last token logits is appropriate for classification tasks where
         the model predicts the class at the final position
       - Argmax along dimension -1 selects the highest probability class
       - Incremental accumulation avoids storing all predictions in memory
       - Breaking early when reaching batch limit enables efficient partial evaluation
       - Returns accuracy as float between 0.0 and 1.0, or None if no data

       Returns:
           float: Classification accuracy ratio (correct_predictions / total_examples)
                  between 0.0 and 1.0, or None if data loader is empty

       Raises:
           RuntimeError: If model inference fails or tensors are on different devices
           ValueError: If target batch doesn't match predicted shape
       """
        if len(self.data_loader) == 0:                                                     # Method will not calculate the accuracy if the length of data_loader is 0
            return None

        self.gpt_model.eval()
        for each_index, (each_input_batch, each_target_batch) in enumerate(self.data_loader):
            if each_index < self.number_of_batches_in_dataloader:
                each_input_batch, each_target_batch = each_input_batch.to(self.device), each_target_batch.to(self.device)
                with no_grad():
                    logits = self.gpt_model(each_input_batch)[:, -1, :]
                biggest_value = argmax(logits, dim= - 1)
                self.numbers_of_examples += biggest_value.shape[0]
                correct_predictions = (biggest_value == each_target_batch).sum().item()
                self.correct_predictions += correct_predictions
            else: break
        return self.correct_predictions / self.numbers_of_examples