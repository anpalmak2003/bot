import requests
from PIL import Image
from transformers import ViltConfig
from transformers import ViltForQuestionAnswering
from transformers import ViltProcessor
import torch
import os


def translate_text(text, target_language='en'):
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=" + target_language + '&dt=t&q=' + text
    response = requests.get(url)
    translation = response.json()[0][0][0]
    return translation

def process(img_path,text):
        image = Image.open(img_path, mode='r', formats=None)
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model.load_state_dict(torch.load('models/new_vqa_model.pth', map_location=torch.device('cpu')))
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        translated_text = translate_text(text)
        encoding = processor(image, translated_text, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = torch.sigmoid(logits).argmax(-1).item()
        answer = model.config.id2label[idx]
        translated_answer = translate_text(translated_text + answer, 'ru')
        quest, ans = translated_answer.split("?")
        ans = ans.strip()
        return ans[0].upper() + ans[1:]
