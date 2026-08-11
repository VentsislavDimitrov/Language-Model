from logging import info
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import no_grad, device
from numpy.typing import NDArray
from numpy import array, append
from calculate_loss_class import CalculateLoss
from converter_class import Converter
from token_generation_class import GenerateTokens
from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from classification_accuracy_class import ClassificationAccuracy


class TrainModel:
    def __init__(self, gpt_model: GPTModel,
                 train_epoch: int,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 evaluate_frequency: int,
                 evaluate_iteration: int,
                 input_text: str,
                 device: device,
                 classification_mode: bool):

        self.gpt_model: GPTModel = gpt_model
        self.train_epoch: int = train_epoch                                                # On how many loops is the module going to be trained?
        self.train_loader: DataLoader = train_loader                                       # Loading the train dataset, used for iterating through its batches and for classification accuracy measurement
        self.validation_loader: DataLoader = validation_loader
        self.optimizer: AdamW = AdamW(self.gpt_model.parameters(),                         # Getting access to gpt_model's neural network so to adjust the parameters
                                      lr=0.0001,                                           # Very small steps (careful, precise learning)
                                      weight_decay=0.3)                                    # Keeps weights from growing too large, preventing the model from "memorizing" instead of "learning"

        self.evaluate_frequency: int = evaluate_frequency                                  # Number of steps between evaluation runs
        self.evaluate_iteration: int = evaluate_iteration                                  # Number of batches to use when computing evaluation metrics

        self.input_text: str = input_text
        self.device = device
        self.classification_mode: bool = classification_mode


        self.losses_from_train_loader: NDArray = array([] ,dtype=float)              # Array tracking training losses over time
        self.losses_from_validation_loader: NDArray = array([], dtype=float)         # Array tracking validation losses over time
        self.tracked_seen_tokens: NDArray = array([], dtype=float)                   # Array tracking tokens seen at each evaluation point
        self.train_accuracies: NDArray = array([], dtype=float)                      # Array tracking training accuracies per epoch
        self.validation_accuracies: NDArray = array([], dtype=float)                 # Array tracking validation accuracies per epoch


        self.seen_tokens: int = 0
        self.global_step: int = -1

        self.converter = Converter()






    def train_model(self):
        """
        Execute the complete training loop for the specified number of epochs.

        Algorithm:
        ==========
        1. For each training epoch:
           a. Set model to training mode
           b. Iterate through each batch in train_loader:
              - Reset optimizer gradients with zero_grad()
              - Compute loss for the batch using CalculateLoss
              - Backpropagate loss with backward()
              - Update model parameters with optimizer.step()
              - Increment seen_tokens and global_step counters
              - At evaluate_frequency intervals: evaluate on train and validation sets
              - Record losses and token counts
           c. After epoch completion: compute classification accuracy on both DATASETS
           d. Log and store accuracy metrics
        2. Return all tracked metrics for analysis

        Why This Design:
        ================
        - Training mode (model.train()) enables dropout and batch normalization behaviors
        - Zero_grad() prevents gradient accumulation from previous batches
        - Loss computation is delegated to CalculateLoss for separation of concerns
        - Periodic evaluation during training provides early insight into model performance
        - Recording losses at intervals (not every step) reduces memory usage and storage
        - Classification accuracy computed per epoch gives holistic view of model performance
        - Combining loss tracking with accuracy measurement provides complementary metrics
        - Returning all tracked arrays enables post-training visualization and analysis
        - info logging provides real-time visibility into training progress

        Returns:
            Tuple containing:
            - losses_from_train_loader: Array of training losses recorded during evaluation
            - losses_from_validation_loader: Array of validation losses recorded during evaluation
            - tracked_seen_tokens: Array of token counts at each evaluation point
            - train_accuracies: Array of training accuracies per epoch
            - validation_accuracies: Array of validation accuracies per epoch
        """
        for each_successful_epoch in range(self.train_epoch):                              # The amount of iterations the training loop needs to be completed
            self.gpt_model.train()                                                         # Starting the training loop for each train_epoch
            for each_input_batch, each_target_batch in self.train_loader:
                self.optimizer.zero_grad()                                                 # Resetting the loss gradient calculation for each iteration
                calculate_loss = CalculateLoss(gpt_model=self.gpt_model,
                                               input_batch=each_input_batch,
                                               target_batch=each_target_batch,
                                               device=self.device,
                                               classification_mode=self.classification_mode)

                loss_for_each_batch = calculate_loss.calculate_loss_of_batches()
                loss_for_each_batch.backward()                                             # Calculating the loss gradient for each batch in train loader
                self.optimizer.step()                                                      # Updating model weights

                if self.classification_mode == True: self.seen_tokens += each_input_batch.shape[0]
                else: self.seen_tokens += each_input_batch.numel()
                self.global_step += 1

                if self.global_step % self.evaluate_frequency == 0:
                    (train_data_loader_loss,
                     validation_data_loader_loss) = self.evaluate_model()                  # Evaluating the model, by calculating the loss of both train and validation loader, if there is no remainders

                    self.losses_from_train_loader = append(self.losses_from_train_loader, train_data_loader_loss)
                    self.losses_from_validation_loader = append(self.losses_from_validation_loader, validation_data_loader_loss)
                    self.tracked_seen_tokens = append(self.tracked_seen_tokens, self.seen_tokens)
                    info(f"Epoch Iteration: {each_successful_epoch + 1} (Step: {self.global_step:06d}):\n"
                                 f"Train Loss: {train_data_loader_loss:.3f}\n"
                                 f"Validation Loss: {validation_data_loader_loss:.3f}")




            if self.classification_mode == True:
                train_classification_accuracy = ClassificationAccuracy(data_loader=self.train_loader,
                                                                       gpt_model=self.gpt_model,
                                                                       device=self.device,
                                                                       number_of_batches=self.evaluate_iteration)
                validation_classification_accuracy = ClassificationAccuracy(data_loader=self.validation_loader,
                                                                            gpt_model=self.gpt_model,
                                                                            device=self.device,
                                                                            number_of_batches=self.evaluate_iteration)

                calculate_train_accuracy = train_classification_accuracy.calculate_accuracy()
                calculate_validation_accuracy = validation_classification_accuracy.calculate_accuracy()

                info(f"Training accuracy: {calculate_train_accuracy*100:.2f}% | \n"
                     f"Validation accuracy: {calculate_validation_accuracy*100:.2f}% | \n")
                self.train_accuracies = append(self.train_accuracies, calculate_train_accuracy)
                self.validation_accuracies = append(self.validation_accuracies, calculate_validation_accuracy)

            else:
                self.generate_and_print_sample()




        if self.classification_mode == True: return self.losses_from_train_loader, self.losses_from_validation_loader, self.tracked_seen_tokens, self.train_accuracies, self.validation_accuracies
        else: return self.losses_from_train_loader, self.losses_from_validation_loader, self.tracked_seen_tokens





    def evaluate_model(self) -> tuple[float, float]:
        """
        Evaluate the model's performance on train and validation DATASETS.

        Algorithm:
        ==========
        1. Set model to evaluation mode (disables dropout, etc.)
        2. Disable gradient computation with no_grad() context manager
        3. Instantiate CalculateLoss for training data loader
        4. Instantiate CalculateLoss for validation data loader
        5. Compute average loss on a subset of batches from each loader
        6. Return model to training mode
        7. Return both losses as a tuple

        Why This Design:
        ================
        - Evaluation mode ensures consistent behavior without stochastic elements
        - no_grad() prevents unnecessary gradient computation, saving memory and compute
        - Uses evaluate_iteration to limit batches for quick, representative evaluation
        - Computes both train and validation losses simultaneously for efficiency
        - Separate loss calculations for each dataset enable overfitting detection:
          if validation loss increases while training loss decreases, model is overfitting
        - Returns losses as tuple for immediate unpacking in calling method
        - Restores training mode to prepare for continued training after evaluation

        Returns:
            Tuple of (train_loss, validation_loss) as floating point values
        """
        self.gpt_model.eval()
        with no_grad():
            calculate_loss_of_train_loader = CalculateLoss(gpt_model=self.gpt_model,
                                                           data_loader=self.train_loader,
                                                           number_of_batches_in_dataloader=self.evaluate_iteration,
                                                           device=self.device,
                                                           classification_mode=self.classification_mode)
            calculate_loss_of_validation_loader = CalculateLoss(gpt_model=self.gpt_model,
                                                                data_loader=self.validation_loader,
                                                                number_of_batches_in_dataloader=self.evaluate_iteration,
                                                                device=self.device,
                                                                classification_mode=self.classification_mode)
            train_data_loader_loss = calculate_loss_of_train_loader.calculate_loss_of_loader()
            validation_data_loader_loss = calculate_loss_of_validation_loader.calculate_loss_of_loader()
        self.gpt_model.train()
        return train_data_loader_loss, validation_data_loader_loss



    def generate_and_print_sample(self):
        """
        Generate and log a sample text from the current model state.

        Algorithm:
        ==========
        1. Set model to evaluation mode
        2. Retrieve model's context length from position embedding size
        3. Convert input_text to token IDs using Converter
        4. Disable gradient computation with no_grad()
        5. Instantiate GenerateTokens with model, encoded input, and generation parameters
        6. Generate new tokens with temperature=0.1 (low randomness) and no top-k filtering
        7. Convert generated token IDs back to text using Converter
        8. Log the generated text with newlines replaced by spaces for compact display
        9. Return model to training mode

        Why This Design:
        ================
        - Evaluation mode ensures deterministic generation (no dropout)
        - Low temperature (0.1) produces more focused, less random text generation
        - Maximal new tokens (50) limits generation length for quick, readable samples
        - Top_k=None allows vocabulary-wide token sampling with temperature scaling
        - Sample generation provides qualitative insight into model learning progress
        - Using no_grad() prevents gradient computation during generation, saving memory
        - Text replacement makes log output cleaner and easier to read
        - Position embedding shape determines context window for generation
        - Restoring training mode prepares model to resume training after sampling

        Note:
            This method is currently commented out in train_model() but remains
            functional for manual inspection during development.
        """
        self.gpt_model.eval()
        context_length = self.gpt_model.position_embedding.weight.shape[0]
        encoded_input = self.converter.text_to_token_ids(text=self.input_text, add_batch_dim=True)
        with no_grad():
            generate_tokens = GenerateTokens(gpt_model=self.gpt_model,
                                             encoded_input=encoded_input,
                                             context_length=context_length,
                                             maximal_new_tokens=50,
                                             temperature=0.1,
                                             top_k=None)

            decoded_text = self.converter.token_ids_to_text(generate_tokens.generate_tokens())
            info(f"{decoded_text.replace("\n", " ")}")

        self.gpt_model.train()




