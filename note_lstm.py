import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(36, input_shape=(16, 18)))
    model.add(layers.LSTM(72, return_sequences=True))
    model.add(layers.LSTM(72, return_sequences=True))
    model.add(layers.Dense(144))
    model.add(layers.Dense(256))

    return model


create_model().summary()