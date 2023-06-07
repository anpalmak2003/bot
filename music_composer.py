import os

import torch
from music_model.transformer import MusicTransformer
import music_model.constants as constants
import numpy as np
import music_model.generation as generation
import music_model.midi_processing as midi_processing
import midi2audio
from music21 import converter, instrument
import sys

batch_size = 1
id2genre = {0: 'classic', 1: 'jazz', 2: 'calm', 3: 'pop', 4: 'hiphop'}

#перобразовываем миди и сохраняем результат
def decode_and_write(generated, genre, out_dir):
    path = ''
    for i, (gen, g) in enumerate(zip(generated, genre)):
        midi = midi_processing.decode(gen, id2genre[g])
        path = f'{out_dir}/gen_{i:>02}_{id2genre[g]}.mid'
        midi.write(f'{out_dir}/gen_{i:>02}_{id2genre[g]}.mid')
    return path


def generate_music(genre):
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    #загружаем модель
    model = MusicTransformer(map_location, n_layers=12, d_model=1024, dim_feedforward=2048, num_heads=16,
                             vocab_size=constants.VOCAB_SIZE, rpr=True).to(map_location).eval()
    model.load_state_dict(torch.load(os.path.join('models/music.pt'), map_location=map_location))

    primer_genre = np.repeat([genre], batch_size)
    primer = torch.tensor(primer_genre)[:, None] + constants.VOCAB_SIZE - 5

    params = {'target_seq_length': 1024, 'temperature': 1.0, 'topk': 40, 'topp': 0.99, 'topp_temperature': 1.0,
              'at_least_k': 1, 'use_rp': False, 'rp_penalty': 0.05, 'rp_restore_speed': 0.7, 'seed': None}

    #получам сгенерированную мелодию
    generated = generation.generate(model, primer, **params)
    generated = generation.post_process(generated, remove_bad_generations=False)

    midi_path = decode_and_write(generated, primer_genre, os.path.join('output/music'))

    s = converter.parse(midi_path)
    os.remove(midi_path)



    s.write('midi', midi_path)
    midi2audio.FluidSynth("font.sf2").midi_to_audio(midi_path, midi_path.replace('.mid', '.wav'))

    return midi_path.replace('.mid', '.wav')
