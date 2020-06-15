import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.Sequential()
    '''
    model.add(layers.TimeDistributed(layers.Dense(64, activation='tanh')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    '''
    model.add(layers.LSTM(128, activation='tanh', stateful=True, return_sequences=True, batch_input_shape=(1, 1, 22)))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(128, activation='tanh', stateful=True, return_sequences=True))
    model.add(layers.Dense(256, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc', tf.keras.metrics.AUC()])

    return model

