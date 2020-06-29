import os

import cv2
import numpy as np
import note_lstm
import c_lstm_model
from my_utils.data_loader import slice_image
from my_utils.note_embedding import note_map
import matplotlib.pyplot as plt

IMAGE_PATH = './test_images/'

#future gazer

id2note, note2id = note_map(4)


def generate_onset(difficulty, od, hd, image_path):
    print("==== create placement set ====")
    placement_model = c_lstm_model.create_model(16, (80, 15, 3), (3,))
    placement_model.load_weights('checkpoints/model_default')

    image_size = len(os.listdir(image_path))

    diff_vector = [[difficulty, od, hd] for _ in range(16)]
    diff_vector = np.array(diff_vector).reshape(-1, 16, 3)

    result = []

    for i in range(image_size):
        if i % 2:
            print("%.2f%%" % (i / image_size * 100))

        image = cv2.imread(image_path + str(i) + ".png", cv2.IMREAD_COLOR)
        part_image = np.array(slice_image(image)).reshape(-1, 16, 80, 15, 3)

        result = result + placement_model.predict([part_image, diff_vector]).reshape(-1, ).tolist()
    print("==== Done! ====")

    plt.figure(figsize=(16, 4))
    plt.grid(True)
    plt.xlabel('time step')
    plt.ylabel('%')
    plt.title('onset predict')
    plt.plot(result)
    plt.xticks([i for i in range(64)])
    plt.xlim([0, 63])
    plt.ylim([0, 1])
    plt.axhline(y=0.5, color='r', linewidth=1)
    plt.show()
    return result


def get_count(note_placement):
    count_list = []
    count = 0
    for item in note_placement:
        if item > 0:
            count_list.append(count)
            count = 0
        else:
            count += 1

    return count_list


def get_count_values(count_list):
    count_max = 5.075173815233827
    count_list = np.log(np.array(count_list) + 1) / count_max
    count_list = count_list.tolist()
    count_list.append(count_max)

    return count_list


def generate_note(beat, diff, od, hd, count_list):
    print("==== create note list ====")
    selection_model = note_lstm.create_model()
    selection_model.load_weights('selection_checkpoints/model_default')

    beat_mean = 373.6436695278145
    beat_std = 99.71674024965917

    beat = (beat - beat_mean) / beat_std
    note_result = []
    cur_note = 0
    for i in range(0, len(count_list) - 1):
        if i % 500:
            print("%.2f%%" % (i / len(count_list) * 100))

        x_temp = [beat, diff, od, hd, count_list[i], count_list[i + 1]]

        note = id2note[cur_note]
        for j in range(4):
            for k in range(4):
                x_temp.append(1.0 if int(note[j]) == k else 0.0)

        result = selection_model.predict(np.array(x_temp).reshape(-1, 1, 22))
        cur_note = np.argmax(result[0])
        #print(id2note[cur_note])
        note_result.append(cur_note)

    print("==== Done! ====")
    return note_result


if __name__ == "__main__":
    beat = 434.782608695652
    sixteenth_note = beat // 4

    diff = 0.7
    od = 0.7
    hd = 0.6

    note_placement = [1 if value > 0.5 else 0 for value in generate_onset(diff, od, hd, IMAGE_PATH)]
    count_list = get_count(note_placement)
    count_values = get_count_values(count_list)

    note_list = generate_note(beat, diff, od, hd, count_values)
    result_list = []

    note_count = 0
    for i in range(len(note_placement) - 1):
        if note_placement[i] == 1:
            result_list.append([int(i * sixteenth_note), id2note[note_list[note_count]]])
            note_count += 1

    result_list.reverse()

    for item in result_list:
        print("%5d: %s"%(item[0], item[1]))
