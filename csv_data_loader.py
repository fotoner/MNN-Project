import json
import pandas as pd


CSV_PATH = "./note_csv/"


def get_song_list():
    with open("processed_list.json", "r", encoding="utf-8") as f:
        song_list = json.load(f)

    return song_list


if __name__ == "__main__":
    song_list = get_song_list()["processed_list"]

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
            while len(note_list) % 16 != 0:
                note_list.append(0)
                count_list.append(count_max)



            break
    '''
    for song in song_list:
        title = song["title"]
        diff_list = song["diff_list"]
    '''

