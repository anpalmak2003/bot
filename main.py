import urllib

import telebot
from telebot import types  # для указание типов
import config
import eggs_classifiaction
import os
import music_composer
import photo_describer
import photo_generation
from gpt import Gpt


bot = telebot.TeleBot(config.token_bot)
cur_state = 'communication'  # music eggs communication photo_gen photo_des
img_pth_to_describe =''
start_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

btn1 = types.KeyboardButton("Оценить качество приготовления яиц")
btn2 = types.KeyboardButton("Сгенерировать мелодию")
btn3 = types.KeyboardButton("Сгенерировать картинку")
  #  btn4 = types.KeyboardButton("Пообщаться")
btn5 = types.KeyboardButton("Задать вопрос по картинке")
start_markup.add(btn1, btn2, btn3, btn5)

replayer=Gpt()



def save_image(file_info, image_name=None):
    downloaded_file = bot.download_file(file_info.file_path)

    if image_name:
        image_path = os.path.join('input', 'photos', image_name + '.jpg')
    else:
        image_path = os.path.join('input', file_info.file_path)

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    return image_path


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    global bot_state, content_image_path,img_pth_to_describe
    if cur_state == 'eggs':
        file_info = bot.get_file(message.photo[-1].file_id)
        image_path = save_image(file_info, 'eggs')

        sout = eggs_classifiaction.process_img(image_path)
        bot.send_message(message.chat.id,
                         text=sout.format(
                             message.from_user), reply_markup=start_markup)
    if cur_state == 'photo_des':
        file_info = bot.get_file(message.photo[-1].file_id)
        img_pth_to_describe = save_image(file_info, 'describe')
        bot.send_message(message.chat.id,
                         text='Задайте вопрос по картинке')




@bot.message_handler(commands=['start'])  # создаем команду
def start(message):

    bot.send_message(message.chat.id,
                     text="Привет, {0.first_name}! Выбери, что ты хочешь или напиши мне...".format(
                         message.from_user), reply_markup=start_markup)


@bot.message_handler(content_types='text')
def message_reply(message):
    global cur_state, markup

    if message.text == "Оценить качество приготовления яиц":
        cur_state = 'eggs'
        bot.send_message(message.chat.id, 'Отправьте фотографию вашей яичницы', reply_markup=markup)

    elif message.text == "Сгенерировать мелодию":
        cur_state = 'music'
        btn1 = types.KeyboardButton("Классика")
        btn2 = types.KeyboardButton("Джазз")
        btn3 = types.KeyboardButton("Спокойная")
        btn4 = types.KeyboardButton("Поп")
        btn5 = types.KeyboardButton("ХипХоп")
        mus_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        mus_markup.add(btn1, btn2, btn3, btn4, btn5)
        bot.send_message(message.chat.id,
                         text="Выберите жанр мелодии".format(
                             message.from_user), reply_markup=mus_markup)

    elif message.text == "Сгенерировать картинку":
        cur_state = 'photo_gen'
        bot.send_message(message.chat.id,
                         text="Введите описание картинки".format(
                             message.from_user), reply_markup=markup)
    elif message.text == "Задать вопрос по картинке":
        cur_state = 'photo_des'
        bot.send_message(message.chat.id,
                         text="Отправьте картинку".format(
                             message.from_user), reply_markup=markup)




    elif (cur_state == 'music'):
        num_genre = 0
        if message.text == "Классика":
            num_genre = 0

        elif message.text == "Джазз":
            num_genre = 1
        elif message.text == "Спокойная":
            num_genre = 2
        elif message.text == "Поп":
            num_genre = 3
        elif message.text == "ХипХоп":
            num_genre = 4

        bot.send_message(message.chat.id, 'Подождите немного пока я сочиню мелодию...', reply_markup=markup)
        music_path = music_composer.generate_music(num_genre)
        bot.send_audio(message.chat.id, audio=open(music_path, 'rb'),reply_markup=start_markup)
        os.remove(music_path)
        os.remove(music_path.replace('wav', 'mid'))
        cur_state == 'communication'

    elif (cur_state == 'photo_gen'):
        photo_name = message.text
        img = photo_generation.generate_photo(photo_name)
        bot.send_message(message.chat.id, 'Подождите немного пока нарисую картинку...', reply_markup=markup)

        bot.send_photo(message.chat.id, urllib.request.urlopen(img).read(), reply_markup=start_markup)
        cur_state == 'communication'

    elif (cur_state=='photo_des'):
        question=message.text
        bot.send_message(message.chat.id, 'Сейчас подумаю...', reply_markup=markup)
        ans=photo_describer.process(img_pth_to_describe,question)

        bot.send_message(message.chat.id, ans, reply_markup=start_markup)
        cur_state == 'communication'
    elif (cur_state=='communication'):
        messg=message.text
        rep=replayer.generate(messg)
        bot.send_message(message.chat.id, rep, reply_markup=start_markup)



bot.infinity_polling()
