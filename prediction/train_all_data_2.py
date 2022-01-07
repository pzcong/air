# 整合0~index规则数据，带有归一化处理
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers import LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential

data_size = 1300
need_num = 100
training_num = 1124
epoch = 30
batch_size = 64


def visualize(x1, x2, rule):
    plt.plot(x1, color='red', label='Truth')
    plt.plot(x2, color='blue', label='Prediction')
    plt.title(label='rule:' + str(rule))
    plt.xlabel(xlabel='Time')
    plt.ylabel(ylabel='Truth')
    plt.xlim([0, 1301 - training_num])  # x轴边界
    plt.xticks(range(1301 - training_num))  # 设置x刻度
    # plt.ylim([0, 15000])  # y轴边界
    # plt.yticks(range(0, 15000, 1000))  # 设置y刻度
    plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='x')
    plt.legend()
    plt.show()


def model_train(rule_i, rule_j):
    # 数据加载
    dataset = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[rule_i:rule_j, :].values
    x_train = []
    y_train = []
    x_validation = []
    sc = MinMaxScaler(feature_range=(0, 1))

    # 训练集预处理
    for l in range(0, rule_j-rule_i):
        training_dataset = dataset[l][:training_num]
        training_dataset = np.append(training_dataset, [[0], [10000]])
        training_dataset_scaled = sc.fit_transform(training_dataset.reshape(-1, 1))
        for i in range(2):
            training_dataset_scaled = np.delete(training_dataset_scaled, obj=training_dataset_scaled.shape[0]-1, axis=0)
        for i in range(need_num, training_dataset_scaled.shape[0]):
            x_train.append(training_dataset_scaled[i-need_num: i])
            y_train.append(training_dataset_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    np.random.seed(1024)
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 验证集预处理
    for l in range(0, rule_j-rule_i):
        inputs = dataset[l][training_num - need_num:]
        inputs = np.append(inputs, [[0], [10000]])
        inputs = sc.fit_transform(inputs.reshape(-1, 1))
        for i in range(2):
            inputs = np.delete(inputs, obj=inputs.shape[0] - 1, axis=0)
        for i in range(need_num, inputs.shape[0]):
            x_validation.append(inputs[i-need_num:i, 0])
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))

    # 模型设置
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=[x_train.shape[1], 1]))
    model.add(BatchNormalization())
    model.add(LSTM(units=128))
    model.add(BatchNormalization())
    model.add(Dense(units=1, activation='relu'))
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 模型训练
    # model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, validation_split=0.01, shuffle=True)
    model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
    # 模型保存
    model.save('model_all_data_fix.h5')

    # 数据预测
    real = []
    for l in range(0, rule_j-rule_i):
        real.extend(dataset[l][training_num:])
    predict = model.predict(x_validation)
    predict = sc.inverse_transform(predict)

    # 预测数据保存
    (pd.Series(real)).to_csv("C:\\Users\\cpz\\Desktop\\real_test.csv", header=False, index=False)
    (pd.DataFrame(predict.astype(int))).to_csv("C:\\Users\\cpz\\Desktop\\predict_test.csv", header=False, index=False)

    # 结果可视化
    for l in range(0, rule_j-rule_i):
        visualize(real[l*(data_size-training_num):(l+1)*(data_size-training_num)], predict[l*(data_size-training_num):(l+1)*(data_size-training_num)], rule_i+l)


if __name__ == '__main__':
    model_train(0, 5)
