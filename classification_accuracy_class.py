from torch.utils.data import DataLoader
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from torch import no_grad,argmax


class ClassificationAccuracy:
    """
    Initialize the ClassificationAccuracy utility.

    Algorithm:
    ==========
    1. Store data loader, GPT model, and device references
    2. Determine the number of batches to evaluate:
       - If number_of_batches is None, use all batches in data_loader
       - Otherwise, take the minimum of specified number and total batches
    3. Initialize counters for correct predictions and total examples

    Why This Design:
    ================
    - Storing references avoids repeated passing of parameters
    - Flexible batch limiting enables quick partial evaluation during training
    - Using min() prevents IndexError when specified batches exceed available batches
    - Counters are initialized to zero and incremented during calculate_accuracy()
    - This class is only instantiated when classification_mode=True in the training loop

    Args:
        data_loader: DataLoader providing batches of (input_ids, target_labels)
        gpt_model: GPTModel instance configured with classification_mode=True
        device: PyTorch device (cpu/cuda) for tensor placement
        number_of_batches: Optional limit on number of batches to evaluate.
                          If None, all batches are used.
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
       This method is specifically designed for use when classification_mode=True.

       Algorithm:
       ==========
       1. Check if data loader is empty; return None if so
       2. Set model to evaluation mode (disables dropout, etc.)
       3. Iterate through batches up to the specified number
       4. Move batch data to the configured device
       5. Use no_grad() context for memory-efficient inference
       6. Forward pass to get logits for all tokens
       7. Extract logits for the last token position (classification token position)
       8. Apply argmax to get predicted class IDs (0 or 1 for binary classification)
       9. Accumulate total examples and correct predictions
       10. Calculate and return accuracy as (correct / total)

       Why This Design:
       ================
       - Empty data loader check prevents division by zero errors
       - Evaluation mode (eval()) ensures consistent behavior vs. training mode
       - Moving data to device handles both CPU and GPU execution seamlessly
       - no_grad() context is essential for memory efficiency during inference
       - Using last token logits is appropriate for classification tasks where
         the model predicts the class at the final position (sentiment, topic, etc.)
       - In classification_mode=True, the output head has 2 units for binary classification
       - Argmax along dimension -1 selects the highest probability class (0 or 1)
       - Incremental accumulation avoids storing all predictions in memory
       - Breaking early when reaching batch limit enables efficient partial evaluation
       - Returns accuracy as float between 0.0 and 1.0, or None if no data
       - This method is called at the end of each epoch when classification_mode=True
         to track model performance on both training and validation sets

       Returns:
           float: Classification accuracy ratio (correct_predictions / total_examples)
                  between 0.0 and 1.0, or None if data loader is empty

       Raises:
           RuntimeError: If model inference fails or tensors are on different devices
           ValueError: If target batch doesn't match predicted shape

       Note:
           This method assumes the model was initialized with classification_mode=True.
           If classification_mode=False, the output head would have vocab_size units
           and this accuracy calculation would not be meaningful.
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