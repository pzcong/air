# 分条训练，不带有归一化，错误模型
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers import LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model

need_num = 100
training_num = 1216
epoch = 50
batch_size = 32


def model_train(rule_i):
    dataset = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input.csv").iloc[rule_i, :].values
    training_dataset = dataset[:training_num]
    sc = MinMaxScaler(feature_range=(0, 1))
    training_dataset_scaled = sc.fit_transform(training_dataset.reshape(-1, 1))
    x_train = []
    y_train = []
    x_validation = []

    for i in range(need_num, training_dataset_scaled.shape[0]):
        x_train.append(training_dataset_scaled[i-need_num: i])
        y_train.append(training_dataset_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    inputs = dataset[training_num-need_num:]
    inputs = sc.fit_transform(inputs.reshape(-1, 1))
    for i in range(need_num, inputs.shape[0]):
        x_validation.append(inputs[i-need_num:i, 0])
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))
    if rule_i == 0:
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=[x_train.shape[1], 1]))
        model.add(BatchNormalization())
        model.add(LSTM(units=128))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
    else:
        model = load_model("model_set.h5")
        model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)

    model.save('model_set.h5')
    real = dataset[training_num:]
    predict = model.predict(x_validation)
    predict = sc.inverse_transform(predict)

    plt.plot(real, color='red', label='Truth')
    plt.plot(predict, color='blue', label='Prediction')
    plt.title(label='Prediction'+str())
    plt.xlabel(xlabel='Time')
    plt.ylabel(ylabel='Truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for k in range(8563):
        model_train(k)