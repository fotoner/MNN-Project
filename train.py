import c_lstm_model
from my_utils.data_loader import data_generator

model = c_lstm_model.create_model(16, (80, 15, 3), (3,))

y_bat = []
result = []


for x_image_batch, x_diff_batch, y_batch in data_generator(batch_size=8192, epochs=5):
    model.fit([x_image_batch, x_diff_batch], y_batch)

model.save_weights('checkpoints/model_default')