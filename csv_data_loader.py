import json
import random

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from my_utils.note_embedding import note_map

CSV_PATH = "./note_csv/"


def get_song_list():
    with open("processed_list.json", "r", encoding="utf-8") as f:
        song_list = json.load(f)

    return song_list


def load_csv_data():
    id2note, note2id = note_map(4)
    song_list = get_song_list()["processed_list"]
    random.shuffle(song_list)
    beat_mean = 373.6436695278145
    beat_std = 99.71674024965917

    count_min = 0.0
    count_max = 5.075173815233827

    train_x = []
    train_y = []

    for song in song_list:
        title = song["title"]
        diff_list = song["diff_list"]
        beat = (song['beat'] - beat_mean) / beat_std

        for diff in diff_list:
            count_list = []
            note_list = []

            difficulty_rating = float(diff["difficulty_rating"]) / 10
            diff_overall = float(diff["diff_overall"]) / 10
            diff_drain = float(diff["diff_drain"]) / 10

            df = pd.read_csv(CSV_PATH + title + '/' + diff['diff'] + '.csv')
            count = 0
            for item in df.values:
                if item[1] > 0:
                    note_list.append(item[1])
                    count_list.append(count)
                    count = 0
                else:
                    count += 1

            count_list = np.log(np.array(count_list) + 1) / count_max
            count_list = count_list.tolist()

            while len(note_list) % 16 != 0:
                note_list.append(0)
                count_list.append(count_max)

            count_list.append(count_max)
            y_set = to_categorical(note_list, num_classes=256).reshape(-1, 1, 256)
            x_set = []
            for i in range(0, len(count_list) - 1):
                x_temp = [beat, difficulty_rating, diff_overall, diff_drain, count_list[i], count_list[i + 1]]

                note = id2note[note_list[i - 1]]
                for j in range(4):
                    for k in range(4):
                        x_temp.append(1.0 if int(note[j]) == k else 0.0)

                x_set.append(x_temp)

            train_x.append(np.array(x_set).reshape(-1, 1, 22))
            train_y.append(y_set)

    return train_x, train_y
