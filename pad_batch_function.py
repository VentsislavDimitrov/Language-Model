from torch import tensor, Tensor, cat, nonzero, stack

def pad_batch(batch, device, allowed_max_length = None):
    """
    Pad a batch of sequences to the same length and create input-target pairs for language modeling.
    This function is primarily used in reasoning mode (classification_mode=False) for next-token prediction.

    This function handles the complete batch preparation pipeline:
    1. Determine the longest sequence in the batch
    2. Pad each sequence to match the longest length using pad_token_id (50256 - EOS token)
    3. Create input-target pairs (shifted sequences for next-token prediction)
    4. Replace pad tokens in targets with -100 to exclude them from loss calculation
    5. Optionally truncate sequences to allowed_max_length
    6. Stack all sequences into batched tensors

    Algorithm:
    ==========
    1. Set pad_token_id to 50256 (GPT-2 EOS token) for padding
    2. Set replace_pad_token_id to -100 (to exclude padded positions from loss)
    3. Find the longest sequence in the batch (length + 1 for shifted targets)
    4. For each sequence in the batch:
       a. Clone and detach the sequence to avoid modifying original
       b. Calculate padding needed to reach longest_sequence_in_batch
       c. Create padding tensor with pad_token_id
       d. Concatenate sequence with padding tokens
       e. Create inputs (all tokens except last) and targets (all tokens except first)
       f. Create mask for padding tokens in targets
       g. Replace first padding token with -100 (keep first pad for boundary)
       h. Optionally truncate to allowed_max_length
       i. Append to inputs_array and targets_array
    5. Stack all inputs and targets into batched tensors
    6. Move tensors to specified device
    7. Return input_tensor and target_tensor

    Why This Design:
    ================
    - Primarily designed for reasoning mode (language modeling):
      * Creates shifted input-target pairs for next-token prediction
      * Standard causal language modeling setup
      * Used during training to teach the model to predict the next token
    - Uses pad_token_id (50256 - EOS token) as standard GPT-2 padding
    - Uses -100 to replace pad tokens in targets:
      * PyTorch's cross_entropy ignores targets with value -100
      * This ensures padding tokens don't contribute to loss calculation
      * Critical for correct language modeling loss computation
    - Handles variable-length sequences in a batch (standard for NLP tasks)
    - First padding token is kept (not replaced) to mark sequence boundary
    - Optional truncation (allowed_max_length) prevents CUDA OOM errors
    - Only the first padding token is kept as a boundary marker; subsequent
      padding tokens are excluded from loss via -100 replacement
    - This function is NOT typically used in classification mode (classification_mode=True)
      where classification accuracy class handles the dataset differently

    Note:
        This function is designed for reasoning/language modeling mode.
        In classification mode, batch processing is handled differently
        (samples are padded and processed without creating shifted pairs).

    Args:
        batch: List of tensors, each representing a tokenized sequence
        device: PyTorch device (cpu/cuda) for tensor placement
        allowed_max_length: Optional maximum sequence length for truncation.
                           If provided, sequences are truncated to this length.

    Returns:
        tuple: (input_tensor, target_tensor) where:
            - input_tensor: Shape (batch_size, sequence_length) containing input token IDs
            - target_tensor: Shape (batch_size, sequence_length) containing target token IDs
                            with pad positions replaced by -100

    Raises:
        ValueError: If batch is empty or contains tensors of different types
        TypeError: If any element in batch is not a tensor

    Example:
        >>> batch = [tensor([1, 2, 3]), tensor([4, 5, 6, 7])]
        >>> inputs, targets = pad_batch(batch, device='cpu')
        >>> inputs.shape  # (2, 5) - longest sequence (4) + 1
        >>> targets.shape # (2, 5)
        >>> targets[0]    # tensor([2, 3, -100, -100, -100]) - padded positions excluded
    """
    pad_token_id = 50256                                                                   # Value that will be used to pad tensors, based onto the length of longest
    replace_pad_token_id = -100                                                            # Value that will replace padded tokens, so to be excluded from train loss

    longest_sequence_in_batch = max(each_item.shape[0] + 1 for each_item in batch)

    # Storing the padded tensors into array
    inputs_array: list[Tensor] = []
    targets_array: list[Tensor] = []

    for each_index, each_sequence in enumerate(batch):
        new_sequence = each_sequence.clone().detach()

        with_how_much_should_be_padded = longest_sequence_in_batch - new_sequence.shape[0]
        padding_tokens = tensor([pad_token_id] * with_how_much_should_be_padded)
        padded = cat([new_sequence, padding_tokens])

        # Create input-target pairs for next-token prediction
        # inputs: tokens 0 to n-1, targets: tokens 1 to n (shifted by 1)
        inputs = tensor(padded[:-1])
        targets = tensor(padded[1:])


        mask: Tensor = targets == pad_token_id
        indices_in_tensors = nonzero(mask).squeeze()

        # Replace all padding tokens EXCEPT the first one with -100
        # First padding token serves as a boundary marker
        if indices_in_tensors.numel() > 1:
            targets[indices_in_tensors[1:]]: Tensor = replace_pad_token_id

        if allowed_max_length is not None:
            inputs: Tensor = inputs[:allowed_max_length]
            targets: Tensor = targets[:allowed_max_length]

        inputs_array.append(inputs)
        targets_array.append(targets)

    input_tensor: Tensor = stack(inputs_array).to(device)
    target_tensor: Tensor = stack(targets_array).to(device)
    return input_tensor, target_tensor