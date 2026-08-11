import pandas
import torch
from torch.utils.data import Dataset
from torch import tensor, Tensor, long, bool

class EncodeAndPad(Dataset):
    """
    Initialize the EncodeAndPad dataset with CSV data and tokenizer.

    Algorithm:
    ==========
    1. Set the pad token ID to 50256 (GPT-2's <|endoftext|> token)
    2. Read the CSV file using pandas to access 'Text' and 'Label' columns
    3. Encode each text entry using the provided tokenizer
    4. Determine max_length:
       a. If None, use the longest encoded sequence in the dataset
       b. If specified, use the provided value and truncate sequences
    5. Pad all encoded sequences to max_length using the pad token ID

    Why This Design:
    ================
    - 50256 is the standard padding token ID for GPT-2, ensuring compatibility
    - Pandas read_csv provides efficient file I/O and column access
    - List comprehension offers fast, readable encoding of all texts
    - Dynamic max_length calculation adapts to dataset without requiring manual tuning
    - User-specified max_length allows consistency across different DATASETS
    - Truncation before padding prevents sequences longer than model's context window
    - Padding after truncation ensures all sequences are exactly max_length long
    - Storing encoded texts in a list enables fast indexing during __getitem__
    - Separating encoding and padding from retrieval improves initialization speed

    Args:
        csv_file: Path to the CSV file containing 'Text' and 'Label' columns
        tokenizer: Tokenizer object with encode() method for text to token IDs
        max_length: Optional maximum sequence length. If None, uses longest sequence

    Raises:
        FileNotFoundError: If csv_file does not exist
        KeyError: If CSV does not contain 'Text' or 'Label' columns
        ValueError: If max_length is less than 1
        TypeError: If tokenizer doesn't have an encode() method
    """
    def __init__(self, csv_file, tokenizer, max_length = None):
        self.pad_token_id: int = 50256
        self.read_data = pandas.read_csv(csv_file)                                         # Reading the selected CSV file
        self.encoded_texts: list = [tokenizer.encode(each_text_line)
                                    for each_text_line in self.read_data["Text"]]          # Each text in "Text" column is being encoded

        if max_length is None:                                                             # If specific length is not set
            self.max_length = self.longest_encoded_length()                                # Calling the longest encoded length method to find the text with the longest length
        else:
            self.max_length = max_length                                                   # Setting the specific maximal length for padding
            self.encoded_texts: list = [each_encoded_text[:self.max_length]
                                        for each_encoded_text in self.encoded_texts]       # Setting the length of each encoded text, based onto the inputted maximal length

        self.encoded_texts = [encoded_text + [self.pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]


    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        """
        Retrieve the encoded text and corresponding label at the given index.

        Algorithm:
        ==========
        1. Get the encoded token sequence at the specified index
        2. Retrieve the corresponding label from the CSV data
        3. Convert both to PyTorch tensors with long dtype (int64)

        Why This Design:
        ================
        - Returns tensors directly for seamless integration with PyTorch training loops
        - Uses long dtype (int64) as required for token IDs and class labels
        - Indexing into encoded_texts list is O(1) for fast retrieval
        - iloc indexing provides positional access to DataFrame rows
        - Returns a tuple of (input_tensor, target_tensor) matching standard dataset format
        - Tuple return type hint improves code clarity and IDE support

        Args:
            index: Integer index of the sample to retrieve

        Returns:
            tuple: (encoded_text_tensor, label_tensor) where:
                   - encoded_text_tensor: Tensor of shape (max_length,) with token IDs
                   - label_tensor: Tensor containing the class label

        Raises:
            IndexError: If index is out of range
        """
        encoded_text = self.encoded_texts[index]
        label = self.read_data.iloc[index]["Label"]
        return tensor(encoded_text, dtype=long),tensor(label, dtype=long)

    def __len__(self) -> int:
        """
       Return the total number of samples in the dataset.

       Algorithm:
       ==========
       1. Return the length of the pandas DataFrame

       Why This Design:
       ================
       - Required by PyTorch Dataset for iteration and batch sampling
       - DataFrame length matches number of rows in CSV, representing dataset size
       - Simple, O(1) operation that doesn't require iterating through encoded texts
       - Enables DataLoader to know how many batches to create

       Returns:
           int: Total number of samples in the dataset
       """
        return len(self.read_data)

    def longest_encoded_length(self) -> int:
        """
        Find the maximum length among all encoded sequences.

        Algorithm:
        ==========
        1. Initialize max_length to 0
        2. Iterate through all encoded text sequences
        3. For each sequence, check if its length exceeds current max_length
        4. Update max_length if a longer sequence is found
        5. Return the maximum length found

        Why This Design:
        ================
        - Used for automatic max_length determination when not user-specified
        - Avoids padding to unnecessarily long sequences if dataset is small
        - Manual loop is more memory efficient than using max(len(x) for x in list)
        - O(n) time complexity where n is the number of sequences
        - Ensures no information loss by keeping all sequences at their original length
        - Called only once during initialization for efficiency
        - Returns integer length suitable for tensor creation

        Returns:
            int: The length of the longest encoded sequence in the dataset
        """
        max_length = 0
        for each_row in self.encoded_texts:
            longest = len(each_row)
            if longest > max_length:
                max_length = longest

        return max_length

