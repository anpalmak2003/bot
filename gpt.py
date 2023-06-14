from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch


class Gpt():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('arood0/model_ru_gpt')
        self.model = GPT2LMHeadModel.from_pretrained('arood0/final_model_gpt_ru')

    def generate(self, input_ids, end_token):
        caption = self.model.generate(input_ids=input_ids,
                                      max_length=500,
                                      min_length=input_ids.shape[1] + 1,
                                      temperature=0.95,
                                      top_k=5000,
                                      top_p=0.98,
                                      eos_token_id=end_token,
                                      pad_token_id=50257,
                                      repetition_penalty=1.2,
                                      bad_words_ids=[[50261]]
                                      )

        return caption, self.tokenizer.decode(caption[0][input_ids.shape[1]:-1])



