import json
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from my_utils.note_embedding import note_map

CSV_PATH = "./note_csv/"


def get_song_list():
    with open("processed_list.json", "r", encoding="utf-8") as f:
        song_list = json.load(f)

    return song_list


if __name__ == "__main__":
    song_list = get_song_list()["processed_list"]
    id2note, note2id = note_map(4)

    song = song_list[0]
    title = song["title"]
    diff_list = song["diff_list"]

    count_list = []

    for song in song_list:
        title = song["title"]
        diff_list = song["diff_list"]
        for diff in diff_list:
            df = pd.read_csv(CSV_PATH + title + '/' + diff['diff'] + '.csv')
            count = 0
            for item in df.values:
                if item[1] > 0:
                    count_list.append(count)
                    count = 0
                else:
                    count += 1

    count_list = np.log(np.array(count_list) + 1)
    count_min = np.min(count_list)
    count_max = np.max(count_list)
    count_norm = (count_list - count_min) / (count_max - count_min)
    count_list = count_norm.tolist()
    count_list.sort()
    plt.plot(count_list)
    plt.show()
    '''
    for song in song_list:
        title = song["title"]
        diff_list = song["diff_list"]
    '''

