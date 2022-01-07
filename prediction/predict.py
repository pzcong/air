import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import copy
import time

need_num = 100
training_num = 1124
epoch = 30
batch_size = 64


def visualize(x1, x2, rule):
    plt.plot(x1, color='orangered', label='Truth')
    plt.plot(x2, color='dodgerblue', label='Prediction')
    # plt.title(label='rule:' + str(rule))
    plt.xlabel(xlabel='Period')
    plt.ylabel(ylabel='Value')
    plt.xlim([0, 180])  # x轴边界
    plt.xticks(range(0, 181, 20))  # 设置x刻度
    # plt.ylim([0, 15000])  # y轴边界
    # plt.yticks(range(0, 15000, 1000))  # 设置y刻度
    # plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='x')
    plt.legend()
    plt.show()


def pre(dataset, model, index):
    # dataset = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[index, :].values
    sc = MinMaxScaler(feature_range=(0, 1))

    x_validation = []
    inputs = dataset[training_num - need_num:]
    inputs = sc.fit_transform(inputs.reshape(-1, 1))

    for i in range(need_num, inputs.shape[0]):
        x_validation.append(inputs[i - need_num:i, 0])
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))

    # real = dataset[training_num:]
    predict = model.predict(x_validation)
    predict = sc.inverse_transform(predict)
    predict = predict.astype(int)

    # 可视化
    '''
    if (index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) or (index <= 100 and index % 20 == 0) or (index % 100 == 0):
        real = dataset[training_num:]
        visualize(real, predict, index)
    '''
    real = dataset[training_num:]
    visualize(real, predict, index)
    return predict


if __name__ == '__main__':
    res = []
    dataset_all = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None)
    model_main = load_model("model_all_data_fix.h5")
    start_time = time.time()
    for k in range(0, 8598):  # 8598
        tmp = dataset_all.iloc[k, :].values
        tmp = pre(tmp, model_main, k)
        tmp = np.squeeze(tmp, axis=1)
        res.append(tmp)
        print(k)

    end_time = time.time()
    (pd.DataFrame(res)).to_csv("C:\\Users\\cpz\\Desktop\\predict_sorted_result.csv", header=False, index=False)
    print("OK")
