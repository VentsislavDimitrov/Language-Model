GPT2_configuration_124M_parameters_Sample: dict[str, int|float|bool] = \
        {
         "vocabulary_size": 50257,                                                         # The ID for each word in the dataset
         "context_length": 256,                                                            # Amount of words AI can see
         "embedding_dimension": 768,                                                       # How long the vector is for each word
         "number_of_heads": 12,                                                            # Number of attention heads
         "number_of_layers": 12,                                                           # Number of layers
         "dropout_rate": 0.1,                                                              # Dropout rate
         "qkv_bias": False                                                                 # Adding extra numbers into Query-Key-Value weights
        }


GPT2_configuration_124M_parameters_V2: dict[str, int|float|bool] = \
        {
         "vocabulary_size": 50257,                                                         # The ID for each word in the dataset
         "context_length": 1024,                                                           # Amount of words AI can see
         "embedding_dimension": 768,                                                       # How long the vector is for each word
         "number_of_heads": 12,                                                            # Number of attention heads
         "number_of_layers": 12,                                                           # Number of layers
         "dropout_rate": 0.1,                                                              # Dropout rate
         "qkv_bias": True                                                                  # Adding extra numbers into Query-Key-Value weights
        }

GPT2_configuration_355M_parameters: dict[str, int|float|bool] = \
        {
         "vocabulary_size": 50257,                                                         # The ID for each word in the dataset
         "context_length": 1024,                                                           # Amount of words AI can see
         "embedding_dimension": 1024,                                                      # How long the vector is for each word
         "number_of_heads": 16,                                                            # Number of attention heads
         "number_of_layers": 24,                                                           # Number of layers
         "dropout_rate": 0.1,                                                              # Dropout rate
         "qkv_bias": True                                                                  # Adding extra numbers into Query-Key-Value weights
        }