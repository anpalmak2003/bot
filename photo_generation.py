import os
import requests
import urllib.request
import replicate
import config

os.environ['REPLICATE_API_TOKEN'] = config.token_photo_gen


# переводим тест
def translate_text(text, target_language='en'):
    url = config.api_link_photo_gen + target_language + '&dt=t&q=' + text
    response = requests.get(url)
    translation = response.json()[0][0][0]
    return translation


# генерируем картинку
def generate_photo(start_prompt):
    output = replicate.run(
        config.kandinski,
        input={"prompt": translate_text(start_prompt) + ", 4k photo"}
    )
    image_url = output[0]
    return image_url
