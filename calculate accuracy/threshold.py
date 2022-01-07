import pandas as pd

sta_size = 100000


def cal_once(index, threshold, real, predict):  # 输入列index 和阈值，返回命中率和新高频规则列表长度
    real = real[:, [index]]
    predict = predict[:, [index]]
    hit_accuracy = 0
    hit_num = []
    for l in range(len(predict)):
        if predict[l] >= threshold:
            hit_num.append(l)
            hit_accuracy += real[l]
    hit_accuracy = hit_accuracy / sta_size
    return hit_accuracy[0], len(hit_num)


def threshold_set(real, predict, th):  # 计算当前阈值th条件下size和acc的平均值
    cap = 0
    acc = 0
    for i_f in range(176):
        acc_tmp, cap_tmp = cal_once(i_f, th, real, predict)
        cap += cap_tmp
        acc += acc_tmp
    print(str(th))
    print("Cap: " + str(cap / 176))
    print("Acc: " + str(acc / 176))


if __name__ == '__main__':
    tra_pre = pd.read_csv("C:\\Users\\cpz\\Desktop\\predict_sorted_result.csv", header=None).iloc[:].values
    tra_real = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:, 1124:1300].values
    for i in range(1, 21):
        threshold_set(tra_real, tra_pre, i)
    print("OK")
