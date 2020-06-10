import tensorflow as tf
from tensorflow.keras import layers


def difficulty_model():
    '''
    fc1 = layers.Dense(100, activation='relu')
    drop = layers.Dropout(0.3)(fc1)
    fc2 = layers.Dense(200, activation='relu')(drop)
    model = tf.keras.Model(fc1, fc2)
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(200, activation='relu'))


    return model


def cnn_model():
    '''
    conv1 = layers.Conv2D(kernel_size=(3, 7), filters=3, padding='valid', activation='relu')
    pool1 = layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='valid')(conv1)
    conv2 = layers.Conv2D(kernel_size=(3, 3), filters=10, padding='valid', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='valid')(conv2)
    flat = layers.Flatten()(pool2)
    drop = layers.Dropout(0.3)(flat)
    fc = layers.Dense(200, activation='relu')(drop)

    model = tf.keras.Model(conv1, fc)
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(kernel_size=(3, 7), filters=3, padding='valid', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='valid'))
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=10, padding='valid', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(200, activation='relu'))

    return model


def concatenated_model(time_step, cnn_input_shape, difficulty_input_shape):
    cnn_input = layers.Input(shape=(time_step, cnn_input_shape[0], cnn_input_shape[1], cnn_input_shape[2]), name="cnn_input")
    cnn_time = layers.TimeDistributed(cnn_model(), name="cnn_time")(cnn_input)

    diff_input = layers.Input(shape=(time_step, difficulty_input_shape[0]), name="diff_input")
    diff_time = layers.TimeDistributed(difficulty_model(), name="diff_time")(diff_input)

    concatenated_out = layers.concatenate([cnn_time, diff_time])

    return cnn_input, diff_input, concatenated_out


def create_model(time_step, cnn_input_shape, difficulty_input_shape):
    cnn_input, diff_input, concatenated_out = concatenated_model(time_step, cnn_input_shape, difficulty_input_shape)

    do1 = layers.Dropout(0.3)(concatenated_out)
    bi_lstm1 = layers.Bidirectional(layers.LSTM(units=400, return_sequences=True), merge_mode='concat')(do1)
    #lstm1 = layers.LSTM(units=400, return_sequences=True, stateful=True)(do1)
    #lstm2 = layers.LSTM(units=400, return_sequences=True, stateful=True)(lstm1)

    d1 = layers.Dense(400, activation='relu')(bi_lstm1)
    do2 = layers.Dropout(0.3)(d1)

    d2 = layers.Dense(200, activation='relu')(do2)
    do3 = layers.Dropout(0.3)(d2)

    d3 = layers.Dense(100, activation='relu')(do3)
    output = layers.Dense(1, activation='sigmoid')(d3)

    model = tf.keras.Model([cnn_input, diff_input], output)
    model.compile(loss='mse', optimizer='Adam', metrics=['acc', tf.keras.metrics.AUC()])

    return model


if __name__ == "__main__":
    c_lstm = create_model(16, (80, 15, 3), (3,))
