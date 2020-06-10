import json
import random

import numpy as np
import pandas as pd
import cv2
import os

IMG_PATH = "./spected_songs/"
CSV_PATH = "./note_csv/"


def get_song_list():
    with open("./processed_list.json", "r", encoding="utf-8") as f:
        song_list = json.load(f)

    return song_list


def get_song_image_size(title):
    return len(os.listdir(IMG_PATH + title))


def slice_image(image):
    slice_img = []
    for j in range(16):
        img = image[0: 80, j * 15: + (j + 1) * 15] / 255.0
        slice_img.append(img)

    return slice_img


def data_generator(batch_size=2, epochs=1):
    x_image_batch = []
    x_diff_batch = []
    y_batch = []

    for i in range(epochs):
        print("\nEpochs: " + str(i + 1) + " *" * 20)

        song_list = get_song_list()["processed_list"]
        random.shuffle(song_list)
        for song in song_list:
            title = song["title"]
            diff_list = song["diff_list"]
            print(title)

            image_size = get_song_image_size(title)

            x_image_set = []
            for i in range(image_size):
                image = cv2.imread(IMG_PATH + title + "/" + str(i) + ".png", cv2.IMREAD_COLOR)
                part_image = slice_image(image)

                x_image_set.append(part_image)

            for diff in diff_list:
                df = pd.read_csv(CSV_PATH + title + "/" + diff["diff"] + ".csv")
                y_set = (df["note_id"] > 0).astype(int).values.tolist()

                difficulty_rating = float(diff["difficulty_rating"]) / 10
                diff_overall = float(diff["diff_overall"]) / 10
                diff_drain = float(diff["diff_drain"]) / 10

                x_diff_set = [[difficulty_rating, diff_overall, diff_drain] for _ in range(16)]

                for _ in range(image_size - len(y_set) % image_size):
                    y_set.append(0)

                for i in range(len(x_image_set)):
                    y = y_set[:16]
                    if len(y) < 16:
                        break

                    y_set = y_set[16:]
                    x_image = x_image_set[i].copy()

                    x_image_batch.append(x_image)
                    x_diff_batch.append(x_diff_set)
                    y_batch.append(y)

                    if len(y_batch) == batch_size:
                        yield np.array(x_image_batch).astype(np.float32), np.array(x_diff_batch),  np.array(y_batch).reshape(-1, 16, 1)
                        x_image_batch = []
                        x_diff_batch = []
                        y_batch = []

    yield np.array(x_image_batch).astype(np.float32), np.array(x_diff_batch), np.array(y_batch).reshape(-1, 16, 1)