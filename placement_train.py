import c_lstm_model
import matplotlib.pyplot as plt
from my_utils.data_loader import data_generator

model = c_lstm_model.create_model(16, (80, 15, 3), (3,))
#model.load_weights('checkpoints/model_default')

result_auc = []
result_acc = []

for x_image_batch, x_diff_batch, y_batch in data_generator(batch_size=8192,epochs=20):
    hist = model.fit([x_image_batch, x_diff_batch], y_batch, shuffle=True)
    result_auc += hist.history["auc"]
    result_acc += hist.history["acc"]

model.save_weights('checkpoints/model_default')

plt.grid(True)
plt.xlabel('batch step')
plt.ylabel('auc')
plt.title('train result(AUC)')

plt.plot(result_auc)
plt.show()

plt.grid(True)
plt.xlabel('batch step')
plt.ylabel('Accuracy')
plt.title('train result(Accuracy)')

plt.plot(result_acc)
plt.show()
