import re, tiktoken, logging


class Tokenizer:

    def __init__(self):


        self.str_to_int = self.vocabulary()                                                # storing vocabulary for accessing in encode and decode methods
        self.int_to_str = {integer: string for string, integer
                           in self.vocabulary().items()}                                   # inverse vocabulary that turns token ID back into original form


    def vocabulary(self):
        """returns a dictionary of vocabulary with token as key and integer as value"""
        return {token: integer for integer, token in enumerate(self.tokens())}

    def tokenize_input(self, filename: str="dataset.txt"):
        with open(filename, "r", encoding="utf-8") as file:                                # Reading the file
            reading = file.read()
            result = re.split(r'([,.:;?_!"()\']|--|\s)',
                              reading)                                                     # Splitting punctuational signs and whitespaces, in order to be tokenized
            result = [item.strip() for item in result if
                      item.strip()]                                                        # Removing the whitespaces from split sentence
            return sorted(set(result))                                                     # Sorting the text alphabetically, in order to be passed to vocabulary


    def tokens(self):
        """
        Method containing set of dataset from tokenized input,
        along with <|endoftext|> and <|unk|> in case there is unknown word in dataset
        """
        tokens_in_dataset = self.tokenize_input()
        special_tokens = ["<|endoftext|>", "<|unk|>"]
        return tokens_in_dataset + special_tokens

    # def demostrate_data_loader(self):
    #     """Purpose of this method is to show that LLM predict one word at time"""
    #     logging.basicConfig(level=logging.INFO, format="%(message)s")
    #     with open("dataset.txt", "r", encoding="utf-8") as file: raw_text = file.read()
    #     tik_tokenizer = tiktoken.get_encoding("OPENAI_GPT2_124M_WEIGHTS")                                      # Implementing BPE tokenizer
    #     encoded_text = tik_tokenizer.encode(raw_text)
    #     samples = encoded_text[50:]
    #     for i in range(1, 5):
    #         logging.info(f"{tik_tokenizer.decode(samples[:i])}--->"
    #                      f"{tik_tokenizer.decode([samples[i]])}")




    def encode(self, text: str):
        pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', text)                   # Splitting punctuational signs and whitespaces, in order to be tokenized
        pre_processed = [item.strip() for item in pre_processed
                         if item.strip()]                                                  # Removing the whitespaces from split sentence
        pre_processed = [item if item in self.str_to_int else
                         "<|unk|>" for item in pre_processed]                              # In case soecific word does not exist in dataset, the word is replaced with special token
        return [self.str_to_int[string] for string in pre_processed]                       # Converting token into integer

    def decode(self, ids):
        text = " ".join([self.int_to_str[index] for index in ids])                         # Going through each Token ID and joins them with space between the tokens
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)                       # Finds punctuations and replaces them with space






