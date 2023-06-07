import urllib

import telebot
from telebot import types  # для указание типов
import config
import eggs_classifiaction
import os
import music_composer

import photo_generation

bot = telebot.TeleBot(config.token_bot)
cur_state = 'communication'  # music eggs communication photo_gen photo_des


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
    global bot_state, content_image_path
    if cur_state=='eggs':
        file_info = bot.get_file(message.photo[-1].file_id)
        image_path = save_image(file_info, 'eggs')

        sout = eggs_classifiaction.process_img(image_path)
        bot.send_message(message.chat.id,
                         text=sout.format(
                             message.from_user))


@bot.message_handler(commands=['start'])  # создаем команду
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Оценить качество приготовления яиц")
    btn2 = types.KeyboardButton("Сгенерировать мелодию")
    btn3 = types.KeyboardButton("Сгенерировать картинку")
    btn4 = types.KeyboardButton("Пообщаться")
    markup.add(btn1, btn2, btn3, btn4)
    bot.send_message(message.chat.id,
                     text="Привет, {0.first_name}! Выбери, что ты хочешь...".format(
                         message.from_user), reply_markup=markup)


@bot.message_handler(content_types='text')
def message_reply(message):
    global cur_state

    if message.text == "Оценить качество приготовления яиц":
        cur_state = 'eggs'
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id, 'Отправьте фотографию вашей яичницы', reply_markup=markup)

    elif message.text == "Сгенерировать мелодию":
        cur_state = 'music'
        btn1 = types.KeyboardButton("Классика")
        btn2 = types.KeyboardButton("Джазз")
        btn3 = types.KeyboardButton("Спокойная")
        btn4 = types.KeyboardButton("Поп")
        btn5 = types.KeyboardButton("ХипХоп")
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(btn1, btn2, btn3, btn4, btn5)
        bot.send_message(message.chat.id,
                         text="Выберите жанр мелодии".format(
                             message.from_user), reply_markup=markup)

    elif message.text == "Сгенерировать картинку":
        cur_state = 'photo_gen'
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id,
                         text="Введите описание картинки".format(
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
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

        bot.send_message(message.chat.id, 'Подождите немного пока я сочиню мелодию...', reply_markup=markup)
        music_path = music_composer.generate_music(num_genre)
        bot.send_audio(message.chat.id, audio=open(music_path, 'rb'))
        os.remove(music_path)
        os.remove(music_path.replace('wav', 'mid'))
        cur_state == 'communication'

    elif (cur_state == 'photo_gen'):
        photo_name = message.text
        img = photo_generation.generate_photo(photo_name)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id, 'Подождите немного пока нарисую картинку...', reply_markup=markup)

        bot.send_photo(message.chat.id, urllib.request.urlopen(img).read(), reply_markup=markup)
        cur_state == 'communication'


bot.infinity_polling()
