import os
import requests
import urllib.request
import replicate

os.environ['REPLICATE_API_TOKEN'] = 'r8_C68wbNGdJAMAZzRNm2uH1czFIaTOBAc2kdwOk'

def translate_text(text, target_language='en'):
    url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=' + target_language + '&dt=t&q=' + text
    response = requests.get(url)
    translation = response.json()[0][0][0]
    return translation



def generate_photo(start_prompt):
    output = replicate.run(
        "ai-forever/kandinsky-2:601eea49d49003e6ea75a11527209c4f510a93e2112c969d548fbb45b9c4f19f",
        input={"prompt": translate_text(start_prompt) + ", 4k photo"}
    )
    image_url = output[0]

    #file_name = "output/gen_image.jpg"
    #urllib.request.urlretrieve(image_url, file_name)
    return image_url

