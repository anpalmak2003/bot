from kandinsky2 import get_kandinsky2
import requests
import os

'''def save_image(file_info, image_name=None):

    if image_name:
        image_path = os.path.join('input', 'photos', image_name + '.jpg')
    else:
        image_path = os.path.join('input', file_info.file_path)

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    return image_path'''

def translate_text(text, target_language='en'):
    url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=' + target_language + '&dt=t&q=' + text
    response = requests.get(url)
    translation = response.json()[0][0][0]
    return translation



def generate_photo(description):
    model = get_kandinsky2(
        'cuda',
        task_type='text2img',
        cache_dir='/tmp/kandinsky2',
        model_version='2.1',
        use_flash_attention=False
    )
    images = model.generate_text2img(
        description + ", 4k photo",
        num_steps=100,
        batch_size=1,
        guidance_scale=4,
        h=768,
        w=768,
        sampler='p_sampler',
        prior_cf_scale=4,
        prior_steps="20"
    )
    return images[0]

