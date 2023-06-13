from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch

class Gpt():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('arood0/model_ru_gpt')
        self.model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        self.model.resize_token_embeddings(50262)
        self.model.load_state_dict(torch.load("models/model_gpt_ru.ckpt", map_location=torch.device('cpu')))
        self.input_ids = torch.tensor([[50258, 50260]])

    def generate(self, str_in):
        prompt = torch.tensor(self.tokenizer.encode(str_in)).unsqueeze(0).long()
        input_ids = torch.cat([self.input_ids, prompt, torch.tensor([[50261]])], dim=1)
        caption = self.model.generate(input_ids=input_ids,

                                 max_length=128,
                                 min_length=input_ids.shape[1] + 5,
                                 temperature=0.95,
                                 top_k=5000,
                                 top_p=0.98,
                                 eos_token_id=50259,
                                 pad_token_id=50257,
                                 repetition_penalty=1.2,
                                 bad_words_ids=[[50261]]
                                 )
        return  self.tokenizer.decode(caption[0][input_ids.shape[1]:-1])



