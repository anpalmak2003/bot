import torch
import numpy as np
from tqdm import tqdm

import music_model.midi_processing as midi_processing
import music_model.constants as constants
from music_model.midi_processing import RANGES_SUM, get_type, NOTE_ON, NOTE_OFF
from music_model.midi_processing import PIANO_RANGE


def generate(model, primer, target_seq_length=1024, temperature=1.0, topk=40, topp=0.99, topp_temperature=1.0,
             at_least_k=1, use_rp=False, rp_penalty=0.05, rp_restore_speed=0.7, seed=None, **forward_args):
    """

    :param model: модель, которую мы используем для генерации мелодий(обучена заранее)
    :param primer: torch.Tensor (B x N) B-кол-во мелодий, N-длина последовательности, словарь,
    в качестве последних 5 значений включает в себя жанры
    :param target_seq_length: Длина мелодии
    :param temperature: Температура, значения > 1.0 приводят к более стохастической выборке,
    более низкие значения приводят к более ожидаемым и предсказуемым последовательностям
    (в конечном итоге к бесконечно повторяющимся музыкальным паттернам).
    :param topk: Длина набора токенов, из которых будет производиться выборка(более высокие вероятности)
    :param topp: Длина набора токенов, из которых будет производиться выборка(более низкие вероятности)
    :param topp_temperature: Температура для выборки topp
    :param at_least_k: как topk, но заставляют выбирать из >= k токенов с более высокой вероятностью
    :param use_rp: Попытка предотвратить генерацию повторяющихся нот
    :param rp_penalty: Более высокие значения приводят к большему влиянию rp
    :param rp_restore_speed: Как быстро будет снято "наказание" за повторение.
    Более низкие значения приводят к большему влиянию rp
    :param seed:  Исправляет начальное значение для детерминированной генерации.
    :param forward_args: dict, для передачи параметров
    :return: torch.Tensor (B x target_seq_length) сгенерированная последовательность
    """

    device = model.device
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    B, N = primer.shape
    B = 1  # нам нужна только 1 мелодия
    generated = torch.full((B, target_seq_length), constants.TOKEN_PAD, dtype=torch.int64, device=device)
    generated[..., :N] = primer.to(device)

    if use_rp:
        RP_processor = DynamicRepetitionPenaltyProcessor(B, penalty=rp_penalty, restore_speed=rp_restore_speed,
                                                         device=device)
    whitelist_mask = make_whitelist_mask()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(N, target_seq_length)):
            logits = model(generated[:, :i], **forward_args)[:, i - 1, :] # получаем предсказания модели
            logits[:, ~whitelist_mask] = float('-inf')
            p = torch.softmax(logits / topp_temperature, -1)

            # используем topk
            if topk == 0:
                topk = p.shape[-1]
            p_topk, idxs = torch.topk(p, topk, -1, sorted=True)

            # используем topp
            mask = p_topk.cumsum(-1) < topp
            mask[:, :at_least_k] = True
            logits_masked = logits.gather(-1, idxs)
            logits_masked[~mask] = float('-inf')
            p_topp = torch.softmax(logits_masked / temperature, -1)

            # используем use_rp
            if use_rp:
                p_penalized = RP_processor.apply_penalty(p_topp, idxs)
                ib = p_penalized.sum(-1) == 0
                if ib.sum() > 0:
                    # если все токены topp получают нули из-за rp, то возвращаемся к выборке topk
                    p_fallback = p_topk[ib].clone()
                    p_fallback[mask[ib]] = 0.  # zeroing topp
                    p_penalized[ib] = p_fallback

                ib = p_penalized.sum(-1) == 0
                if ib.sum() > 0:
                    # если токены topk получают нули, то возвращаемся к topp без RP
                    print('fallback-2')
                    p_penalized = p_topp
                p_topp = p_penalized

            # берем образец:
            next_token = idxs.gather(-1, torch.multinomial(p_topp, 1))
            generated[:, i] = next_token.squeeze(-1)

            # обновляем penalty:
            if use_rp:
                RP_processor.update(next_token)

    return generated[:, :i + 1]


def post_process(generated, remove_bad_generations=True):
    """
    Функция убирает длинные паузы(более 3 секунд), обрезает скорость, чтобы избежать резко громких нот,
    удаляет плохо сгенерированные образцы(состоящие из повторения одних и тех же нот)

    :param generated: torch.Tensor (B x target_seq_length) сгенерированная последовательность
    :param remove_bad_generations:
    :return: filtered_generated: более чистый и немного лучше звучащий сгенерированная последовательность
    """

    generated = generated.cpu().numpy()
    remove_pauses(generated, 1)
    clip_velocity(generated)

    bad_filter = np.ones(len(generated), dtype=bool)

    if remove_bad_generations:
        for i, gen in enumerate(generated):
            midi = midi_processing.decode(gen)
            if detect_note_repetition(midi) > 0.9:
                bad_filter[i] = False

        if np.sum(bad_filter) != len(bad_filter):
            print(f'{np.sum(~bad_filter)} bad samples will be removed.')

    return generated[bad_filter]


def make_whitelist_mask():
    """Создать маску для PIANO_RANGE"""
    whitelist_mask = np.zeros(constants.VOCAB_SIZE, dtype=bool)
    whitelist_mask[PIANO_RANGE[0]:PIANO_RANGE[1] + 1] = True
    whitelist_mask[128 + PIANO_RANGE[0]:128 + PIANO_RANGE[1] + 1] = True
    whitelist_mask[128 * 2:] = True
    return whitelist_mask


class DynamicRepetitionPenaltyProcessor:

    """
   Класс пытается предотвратить случаи, когда модель генерирует повторяющиеся
   ноты или музыкальные паттерны, ухудшающие качество. Каждая сгенерированная заметка
   будет уменьшать вероятность следующего шага на значение «штрафа» (которое является гиперпараметром)


    bs : int
        количество мелодий(batch_size)
    penalty : float
        значение, на которое вероятность будет уменьшена
    restore_speed : float
        число, обратное количеству секунд, необходимо для полного восстановления вероятности от 0 до 1.
    """

    def __init__(self, bs, device, penalty=0.3, restore_speed=1.0):
        self.bs = bs
        self.penalty = penalty
        self.restore_speed = restore_speed
        self.penalty_matrix = torch.ones(bs, 128).to(device)

    def apply_penalty(self, p, idxs):
        p = p.clone()
        for b in range(len(p)):
            i = idxs[b]
            pi = p[b]
            mask = i < 128
            if len(i) > 0:
                pi[mask] = pi[mask] * self.penalty_matrix[b, i[mask]]
        return p

    def update(self, next_token):
        restoring = next_token - (128 + 128 + 32)
        restoring = torch.clamp(restoring.float(), 0, 100) / 100 * self.restore_speed
        self.penalty_matrix += restoring
        nt = next_token[next_token < 128]
        self.penalty_matrix[:, nt] -= restoring + self.penalty
        torch.clamp(self.penalty_matrix, 0, 1.0, out=self.penalty_matrix)
        return restoring, nt


def detect_note_repetition(midi, threshold_sec=0.01):
    """
    Возвращает долю повторений ноты.
    Подсчитывает случаи, когда prev_note_end == next_note_start с одинаковой высотой звука
     («склеенные» ноты). Используется для обнаружения плохо сгенерированных образцов.

    ----------
    midi : prettyMIDI object
    threshold_sec : float
        интервалы, меньшие порога threshold_sec, рассматриваются как «склеенные» ноты.

    Returns
    -------
    доля повторений нот по отношению к количеству всех нот.
    """
    all_notes = [x for inst in midi.instruments for x in inst.notes if not inst.is_drum]
    if len(all_notes) == 0:
        return 0
    all_notes_np = np.array([[x.start, x.end, x.pitch, x.velocity] for x in all_notes])

    i_sort = np.lexsort([all_notes_np[:, 0], all_notes_np[:, 2]])

    s = []
    cur_p = -1
    cur_t = -1
    for t in all_notes_np[i_sort]:
        a, b, p, v = t
        if cur_p != p:
            cur_p = p
        else:
            s.append(a - cur_t)
        cur_t = b
    s = np.array(s)
    return (s < threshold_sec).sum() / len(s)


def remove_pauses(generated, threshold=3):
    """
   Заполняет паузы значениями const.

    Parameters
    ----------
    generated : torch.Tensor (B x N)
        сгенерированная мелодия
    threshold : int/float
        минимальные секунды тишины, чтобы рассматривать их как паузу.
    """
    mask = (generated >= RANGES_SUM[2]) & (generated < RANGES_SUM[3])
    seconds = ((generated - RANGES_SUM[2]) + 1) * 0.01
    seconds[~mask] = 0

    res_ab = [[] for _ in range(seconds.shape[0])]

    for ib, i_seconds in enumerate(seconds):
        a, s = 0, 0
        notes_down = np.zeros(128, dtype=bool)
        for i, (t, ev) in enumerate(zip(i_seconds, generated[ib])):
            typ = get_type(ev)
            if typ == NOTE_ON:
                pitch = ev
                notes_down[pitch] = True
            if typ == NOTE_OFF:
                pitch = ev - 128
                notes_down[pitch] = False

            if t == 0:
                if s >= threshold and notes_down.sum() == 0:
                    res_ab[ib].append([a, i, s])
                s = 0
                a = i + 1
            s += t
        if s >= threshold and notes_down.sum() == 0:
            res_ab[ib].append([a, len(i_seconds), s])

    # удаление inplace
    for ib, t in enumerate(res_ab):
        for a, b, s in t:
            generated[ib, a:b] = constants.TOKEN_PAD
            print(f'pause removed:', ib, f'n={b - a}', a, b, s)


def clip_velocity(generated, min_velocity=30, max_velocity=100):
    """
    Обрезать скорость до диапазона (min_velocity, max_velocity).
     Поскольку модель иногда генерирует слишком громкие последовательности, мы пытаемся нейтрализовать этот эффект.
    Parameters
    ----------
    generated : torch.Tensor (B x N)
        сгенерированная мелодия
    min_velocity : int
    max_velocity : int
    """
    max_velocity_encoded = max_velocity * 32 // 128 + RANGES_SUM[1]
    min_velocity_encoded = min_velocity * 32 // 128 + RANGES_SUM[1]

    mask = (generated >= RANGES_SUM[1]) & (generated < RANGES_SUM[2])
    generated[mask] = np.clip(generated[mask], min_velocity_encoded, max_velocity_encoded)
