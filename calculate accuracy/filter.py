import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt


def filter_judge(fre_list, index, le):  # 判断是否被过滤，不被返回T,被过滤返回F
    tmp_f = fre_list[index - le: index].astype(int)
    for i_f in range(le):
        if tmp_f[i_f] != 0:
            return True
    return False


def filter_cal_times():
    traffic = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:, 1124:1300].values
    for k_f in range(4, 17):
        filter_num = 0
        for i_f in range(16, 176):
            tmp = 0
            for j_f in range(8598):
                if filter_judge(traffic[j_f], i_f, k_f):
                    tmp += 1
            tmp /= 8598
            filter_num += tmp
        print(str(k_f) + "-Avg:" + str(filter_num / 160))
    print("OK")


def pre_filter_cal_acc(k_f):  # 计算filter的准确率数据预处理，根据真实数据把预测结果中该被过滤的值设为0
    tra_real = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:, 1124:1300].values
    tra_save = pd.read_csv("C:\\Users\\cpz\\Desktop\\predict_sorted_result.csv", header=None).iloc[:].values
    for i_f in range(16, 176):
        for j_f in range(8598):
            if not filter_judge(tra_real[j_f], i_f, k_f):
                tra_save[j_f][i_f] = 0
    (pd.DataFrame(tra_save)).to_csv("C:\\Users\\cpz\\Desktop\\filter_result\\predict_sorted_result_filter_"+str(k_f)+".csv", header=False, index=False)


def cal_once(index, threshold, real, predict):  # 输入列index 和阈值，返回命中率
    sta_size = 100000
    real = real[:, [index]]
    predict = predict[:, [index]]
    hit_accuracy = 0
    for l in range(len(predict)):
        if predict[l] >= threshold:
            hit_accuracy += real[l]
    hit_accuracy = hit_accuracy/sta_size
    return hit_accuracy


def filter_cal_acc(k_f):  # 根据filter文件 计算准确率
    tra_pre_filter = pd.read_csv("C:\\Users\\cpz\\Desktop\\filter_result\\predict_sorted_result_filter_"+str(k_f)+".csv", header=None).iloc[:].values
    tra_real = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:, 1124:1300].values
    acc = 0
    for i_f in range(16, 176):
        acc += cal_once(i_f, 10, tra_real, tra_pre_filter)
    print(str(k_f))
    print(acc/160)


if __name__ == '__main__':
    for i in range(4, 17):
        filter_cal_acc(i)