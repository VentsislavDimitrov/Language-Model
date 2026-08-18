from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from torch import device, tensor, Tensor, no_grad,argmax, cat
from converter_class import Converter


class ClassifyText:
    def __init__(self, text: str,
                 gpt_model: GPTModel, device: device, max_length = None):
        """
        Initialize the ClassifyText utility.

        Algorithm:
        ==========
        1. Store the input text, model, and device references
        2. Store max_length (if provided) for sequence length management
        3. Set pad_token_id to GPT-2's EOS token (50256) for padding
        4. Instantiate Converter for text-token conversion

        Why This Design:
        ================
        - Storing references avoids repeated parameter passing
        - pad_token_id uses EOS token (50256) which is standard for GPT-2 padding
        - max_length can be specified or derived from model's context length
        - Converter is instantiated once for reuse
        - This class is only used when classification_mode=True

        Args:
            text (str): The input text string to classify
            gpt_model (GPTModel): GPTModel instance configured with classification_mode=True
            device (torch.device): PyTorch device (cpu/cuda) for tensor placement
            max_length (int, optional): Maximum sequence length for input.
                If None, will use model's context_length. Defaults to None.
        """
        self.text: str = text
        self.gpt_model: GPTModel = gpt_model
        self.device = device
        self.pad_token_id: int = 50256
        self.max_length = max_length

        self.converter: Converter = Converter()





    def classify_text(self):
        """
        Perform binary classification on the input text.

        Algorithm:
        ==========
        1. Set model to evaluation mode (disables dropout)
        2. Convert input text to token IDs (without batch dimension)
        3. Determine maximum sequence length:
           - If max_length provided: use it
           - Otherwise: use model's context_length from position embedding
        4. Truncate token sequence if it exceeds max_length
        5. Pad token sequence if it's shorter than max_length
           - Pad using pad_token_id (50256 - EOS token)
        6. Add batch dimension with unsqueeze(0)
        7. Move tensor to configured device
        8. Disable gradient computation with no_grad()
        9. Forward pass through model to get logits
        10. Extract logits from the last token position (classification token)
        11. Apply argmax to get predicted class ID (0 or 1)
        12. Convert class ID to human-readable label:
            - 1: "SPAM" (or positive class)
            - 0: "NOT SPAM" (or negative class)

        Why This Design:
        ================
        - Evaluation mode ensures no dropout for consistent predictions
        - Text-to-token conversion without batch dimension allows flexible input
        - Dynamic max_length determination adapts to model's context window
        - Truncation prevents CUDA out-of-memory errors on long inputs
        - Padding ensures all inputs have the same length for batch processing
        - Using EOS token (50256) for padding is standard practice with GPT-2
        - Adding batch dimension enables compatibility with model's forward method
        - no_grad() prevents memory usage from gradient computations
        - Last token extraction: In classification fine-tuning, the model learns
          to output the class prediction at the final token position
        - Argmax selects the highest probability class (0 or 1)
        - Human-readable labels make the output user-friendly
        - This method assumes model was fine-tuned with classification_mode=True

        Returns:
            str: "SPAM" if the model predicts class 1 (positive class),
                 "NOT SPAM" if the model predicts class 0 (negative class)

        Raises:
            RuntimeError: If model inference fails or tensors are on wrong devices
            ValueError: If input text is empty or tokenization fails
            IndexError: If model's output doesn't have the expected shape

        Note:
            This method is designed for inference only and should not be called
            during training. It assumes the model was fine-tuned for binary
            classification with classification_mode=True. The specific labels
            "SPAM" and "NOT SPAM" can be modified to match the actual task
            (e.g., "POSITIVE"/"NEGATIVE" for sentiment analysis).
        """
        self.gpt_model.eval()

        token_ids = self.converter.text_to_token_ids(self.text, add_batch_dim=False)
        supported_text_length = self.gpt_model.position_embedding.weight.shape[1]


        cut_point = min(self.max_length, supported_text_length)
        token_ids = token_ids[:cut_point]
        padding_tensor = tensor([self.pad_token_id] * (self.max_length - len(token_ids)))

        token_ids = cat([token_ids, padding_tensor]).unsqueeze(0)
        input_tensor = token_ids.to(device=self.device)

        with no_grad():
            logits = self.gpt_model(input_tensor)[:, -1]
        predict = argmax(logits, dim = -1).item()
        return "SPAM" if predict == 1 else "NOT SPAM"


