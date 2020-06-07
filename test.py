import os

import cv2
import numpy as np
import c_lstm_model
import matplotlib.pyplot as plt
from my_utils.data_loader import slice_image

PATH = './test_images/'

model = c_lstm_model.create_model(16, (80, 15, 3), (3,))
model.load_weights('checkpoints/model_default')


def generator_onset(difficulty, od, hd, image_path):
    image_size = len(os.listdir(PATH))

    diff_vector = [[difficulty, od, hd] for _ in range(16)]
    diff_vector = np.array(diff_vector).reshape(-1, 16, 3)

    result = []

    for i in range(image_size):
        image = cv2.imread(PATH + str(i) + ".png", cv2.IMREAD_COLOR)
        part_image = np.array(slice_image(image)).reshape(-1, 16, 80, 15, 3)

        result = result + model.predict([part_image, diff_vector]).reshape(-1, ).tolist()

    return result


if __name__ == "__main__":
    plt.figure(figsize=(16, 4))
    plt.grid(True)
    plt.xlabel('time step')
    plt.ylabel('%')
    plt.title('onset predict')
    for diff in range(0, 10, 1):
        onset_raw = generator_onset(diff / 10, 0.7, 0.7, PATH)
        plt.plot(onset_raw[0:64], alpha=0.45)
    plt.xticks([i for i in range(64)])
    plt.xlim([0, 63])
    plt.ylim([0, 1])
    plt.axhline(y=0.5, color='r', linewidth=1)
    plt.show()
