from torch import tensor, Tensor, cat, nonzero, stack

def pad_batch(batch, device, allowed_max_length = None):
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


        inputs = tensor(padded[:-1])
        targets = tensor(padded[1:])


        mask: Tensor = targets == pad_token_id
        indices_in_tensors = nonzero(mask).squeeze()

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