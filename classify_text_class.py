from GPT_ARCHITECTURE_COMPONENTS.GPT import GPTModel
from torch import device, tensor, Tensor, no_grad,argmax, cat
from converter_class import Converter


class ClassifyText:
    def __init__(self, text: str,
                 gpt_model: GPTModel, device: device, max_length = None):

        self.text: str = text
        self.gpt_model: GPTModel = gpt_model
        self.device = device
        self.pad_token_id: int = 50256
        self.max_length = max_length

        self.converter: Converter = Converter()


    def classify_text(self):
        self.gpt_model.eval()

        token_ids = self.converter.text_to_token_ids(self.text, add_batch_dim=False)
        supported_text_length = self.gpt_model.position_embedding.weight.shape[1]


        cut_point = min(self.max_length, supported_text_length)
        token_ids = token_ids[:cut_point]
        padding_tensor = [self.pad_token_id] * (self.max_length - len(token_ids))


        token_ids += padding_tensor
        input_tensor = tensor([token_ids], device=self.device)

        with no_grad():
            logits = self.gpt_model(input_tensor)[:, -1]
        predict = argmax(logits, dim = -1).item()
        return "SPAM" if predict == 1 else "NOT SPAM"


