import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


need_num = 100
training_num = 1124  # 653 # 6500
epoch = 30
batch_size = 64
index = 3

dataset = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv").iloc[index, :].values
training_dataset = dataset[:training_num]
sc = MinMaxScaler(feature_range=(0, 10))
training_dataset = np.append(training_dataset, [[0], [10000]])
training_dataset_scaled = sc.fit_transform(training_dataset.reshape(-1, 1))
for i in range(2):
    training_dataset_scaled = np.delete(training_dataset_scaled, obj=training_dataset_scaled.shape[0]-1, axis=0)
x_train = []
y_train = []
x_validation = []

for i in range(need_num, training_dataset_scaled.shape[0]):
    x_train.append(training_dataset_scaled[i-need_num: i])
    y_train.append(training_dataset_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

inputs = dataset[training_num-need_num:]
inputs = np.append(inputs, [[0], [10000]])
inputs = sc.transform(inputs.reshape(-1, 1))
for i in range(2):
    inputs = np.delete(inputs, obj=inputs.shape[0]-1, axis=0)

for i in range(need_num, inputs.shape[0]):
    x_validation.append(inputs[i-need_num:i, 0])
x_validation = np.array(x_validation)
x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=[x_train.shape[1], 1]))
model.add(BatchNormalization())
model.add(LSTM(units=128))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='relu'))
model.compile(optimizer='adam', metrics=['accuracy'], loss='my_squ_mean_loss')
model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
model.save("model_single_"+str(index)+".h5")

real = dataset[training_num:]
predict = model.predict(x_validation)
predict = sc.inverse_transform(predict)


plt.plot(real, color='red', label='Truth')
plt.plot(predict, color='blue', label='Prediction')
plt.title(label="rule:" + str(index))
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='Truth')
plt.xlim([0, 1301-training_num])  # x轴边界
plt.xticks(range(1300-training_num))  # 设置x刻度
plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='x')
plt.legend()
plt.show()
