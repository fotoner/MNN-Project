import note_lstm
import matplotlib.pyplot as plt
from csv_data_loader import load_csv_data

model = note_lstm.create_model()
#model.load_weights('selection_checkpoints/model_default')
train_x, train_y = load_csv_data()

result_auc = []
result_acc = []
print(len(train_y))
for i in range(5):
    for batch in range(len(train_y)):
        print("================= " + str(i) + " : " + str(batch) + " =================")
        hist = model.fit(train_x[batch], train_y[batch], batch_size=1, shuffle=False)
        result_auc += hist.history["auc"]
        result_acc += hist.history["acc"]

        model.reset_states()

    model.save_weights('selection_checkpoints/model_default')


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
