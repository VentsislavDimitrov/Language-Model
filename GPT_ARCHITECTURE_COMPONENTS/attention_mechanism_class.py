import torch
from torch.nn import Module, Linear, Dropout
from torch import triu, ones, Tensor, inf, softmax, matmul

class AttentionMechanism(Module):

    """
    Self-attention mechanism.

    Module that implements the scaled dot-product attention with multiple attention heads.
    It computes attention weights between all positions in the input sequence and produces
    a context-aware representation for each position.

    Why it's necessary:
    - Allows the model to focus on different parts of the input sequence
    - Enables parallel processing of sequence elements
    - Captures long-range dependencies in sequences
    - Multi-head attention allows focusing on different representation subspaces

    Example:
        >>> attention = AttentionMechanism(
        ...     d_in=512, d_out=512, dropout_range=0.1,
        ...     context_length=100, number_of_heads=8
        ... )
        >>> x = randn(32, 50, 512)  # [batch, seq_len, d_in]
        >>> output = attention(x)  # [32, 50, 512]
    """
    def __init__(self, dimension_input:  int,
                       dimension_output: int,
                       dropout_range:    float,
                       context_length:   int,
                       number_of_heads:  int,
                       qkv_bias:         bool = False):
        assert dimension_output % number_of_heads == 0, "Parameter 'd_out' must be divisible by parameter 'amount_of_heads'"
        super().__init__()

        self.dimension_input:  int   = dimension_input
        self.dimension_output: int   = dimension_output
        self.context_length:   int   = context_length
        self.dropout_range:    float = dropout_range
        self.number_of_heads:  int   = number_of_heads
        self.qkv_bias:         bool  = qkv_bias

        #Reduce the projection dim to match the desired output dimension
        self.dimension_of_head = self.dimension_output // self.number_of_heads


        #D_in and D_out are the dimensions for random weight initialization
        #Creating weight for key/value/query, which will be adjustable
        self.w_key:          Linear = Linear(dimension_input, dimension_output, bias=self.qkv_bias)
        self.w_value:        Linear = Linear(dimension_input, dimension_output, bias=self.qkv_bias)
        self.w_query:        Linear = Linear(dimension_input, dimension_output, bias=self.qkv_bias)


        #This is the output projection layer, which will be adjustable
        self.out_projection: Linear = Linear(dimension_output, dimension_output)


        #Dropping out some of the values in vector with rate 0.5/50%.
        #The dropped out values will be replaced with 0
        self.dropout: Dropout = Dropout(dropout_range)

        #Masking out the attention score, above the diagonal

        #Using PyTorch's buffer will automatically move the tensor to GPU if available

        self.register_buffer('mask',                                                 # <---- Name: "Hey model, remember this as 'mask'"
        triu(ones(context_length,
                              context_length),diagonal=1))                                 # <---- The actual mask




    def masked_attention_score(self,queries, keys, sequence_length):
        """
        Compute scaled dot-product attention scores with masking.

        Why scaled: Dividing by sqrt(d_k) prevents tiny gradients
          when dot products grow large in magnitude.

        Why masked: For autoregressive models, prevents attending to future tokens.

        Args:
              queries: Tensor of shape [batch, heads, seq_len, d_k]
              keys: Tensor of shape [batch, heads, seq_len, d_k]
              sequence_length: Current sequence length (<= context_length)

        Returns:
              Attention scores of shape [batch, heads, seq_len, seq_len] with
              -inf where positions should be masked
        """

        # Finding the attention score for each sentence/prompt

        attention_score = matmul(queries, keys.transpose(2, 3))
        # Masking out the attention score, above the diagonal
        mask = self.mask.bool()[:sequence_length, :sequence_length]
        masked_attention_score = attention_score.masked_fill_(mask, -inf)
        return masked_attention_score

    def cleaned_attention_weight(self,
                         attention_score = None,
                         keys = None) -> Dropout:
        """
        Computing the attention weight matrix and multiplying it with
        the values and changes the position of amount_of_heads and sequence_lengt
        """
        # Renormalizing the attention weight to sum up to 1 for each row
        sqrt_keys_shape = keys.shape[-1] ** 0.5
        attention_weight = softmax(attention_score / sqrt_keys_shape, dim=-1)
        # Applying dropout to the attention weight
        return self.dropout(attention_weight)


    def forward(self, input: Tensor) -> Linear:
        """
        Forward pass through the attention mechanism.

        Steps:
        1. Project input to keys, values, queries
        2. Split into multiple heads
        3. Compute scaled dot-product attention
        4. Apply causal masking
        5. Compute attention weights with softmax
        6. Apply dropout to attention weights
        7. Compute context vectors
        8. Project back to output dimension

        Args:
            input: Input tensor of shape [batch_size, sequence_length, d_in]

        Returns:
            Context-aware representation of shape [batch_size, sequence_length, d_out]
        """
        keys     = self.w_key(input)
        values   = self.w_value(input)
        queries  = self.w_query(input)

        # Splitting the keys, values, and queries into amount of heads
        # number_of_batches: How many sentences we're processing at once
        # sequence_length: How long is each sentence
        # self.amount_of_heads: How many attention heads we're using
        # self.dimension_of_head: How many dimensions each head has

        # Outputs the amount of sentences/prompts, length of sentence, and embedding size
        number_of_batches, sequence_length, embedding_size = input.shape
        keys    = keys.view(number_of_batches,sequence_length,self.number_of_heads,self.dimension_of_head)
        values  = values.view(number_of_batches,sequence_length,self.number_of_heads,self.dimension_of_head)
        queries = queries.view(number_of_batches,sequence_length,self.number_of_heads,self.dimension_of_head)



        # Switching the position of sequence_length and self.number_of_heads dimension
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        mas = self.masked_attention_score(queries, keys, sequence_length)
        caw = self.cleaned_attention_weight(attention_score=mas, keys=keys)


        # Computing the attention weight matrix and multiplying it with
        # the values and changes the position of amount_of_heads and sequence_lengt
        multiply_matrices_for_context_vector
        context_vector = (matmul()).transpose(1, 2)
        context_vector = context_vector.contiguous().view(number_of_batches,
                                                          sequence_length, self.dimension_output)
        return self.out_projection(context_vector)














