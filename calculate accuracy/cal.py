import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
sta_size = 100000


def draw_num(list2, list3):  # 可视化增减数量
    x1 = np.arange(1, 177)
    plt.bar(x1-0.5, list2, width=0.5, color='r', align='edge', label='Add', alpha=1)
    plt.bar(x1, list3, width=0.5, color='b', align='edge', label='Del', alpha=1)
    # plt.axhline(y=np.mean(list2), color='r', ls='--', lw=1, alpha=1)
    # plt.axhline(y=np.mean(list3), color='b', ls=':', lw=1, alpha=1)
    plt.title(label='Quantity Statistics')
    plt.xlabel(xlabel='Period ')
    plt.ylabel(ylabel='Quantity')
    plt.xlim([0, 177])
    plt.ylim([0, 200])
    plt.legend()
    plt.show()


def draw_acc(list1):  # 可视化命中率
    plt.plot(list1, color='blue')
    plt.title(label='Accuracy')
    plt.xlabel(xlabel='Period')
    plt.ylabel(ylabel='Acc(%)')
    plt.xlim([1, 177])  # y轴边界
    plt.ylim([0, 1])  # y轴边界
    plt.yticks(np.arange(0.0, 1.1, 0.1))  # 设置y刻度
    plt.show()


def draw_list_all(list1):  # 可视化热点规则
    plt.plot(list1, color='blue')
    plt.title(label='NUmber of all Hot Rule')
    plt.xlabel(xlabel='Period')
    plt.ylabel(ylabel='Quantity')
    plt.xlim([1, 177])
    plt.ylim([0, 1500])  # y轴边界
    plt.show()


def cal_once(index, threshold, real, predict):  # 输入列index 和阈值，返回命中率和新高频规则列表
    real = real.iloc[:, [index+1124]].values
    predict = predict.iloc[:, [index]].values
    hit_accuracy = 0
    hit_num = []
    for l in range(len(predict)):
        if predict[l] >= threshold:
            hit_num.append(l)
            hit_accuracy += real[l]
    hit_accuracy = hit_accuracy/sta_size
    return hit_accuracy[0], copy.deepcopy(hit_num)


def rule_overlap(list1, list2):  # 求两次规则差集，返回需要插入的规则和需要删除的规则
    return list(set(list2).difference(set(list1))), list(set(list1).difference(set(list2)))


if __name__ == '__main__':
    real_main = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None)
    predict_main = pd.read_csv("C:\\Users\\cpz\\Desktop\\predict_sorted_result.csv", header=None)
    for thr in range(1, 21):  # threshold
        list_last = []
        list_new = []
        Num = []
        Acc = []
        Add_Num = []
        Del_Num = []
        for i in range(176):
            acc, list_new = cal_once(i, thr, real_main, predict_main)
            add, dele = rule_overlap(list_last, list_new)
            Acc.append(acc)
            Num.append(len(list_new))
            Add_Num.append(len(add))
            Del_Num.append(len(dele))
            '''
            print(i)
            print("Acc:"+str(acc))
            print("Num:"+str(len(list_new)))
            print("Add_Num:"+str(len(add)))
            print("Del_Num:" + str(len(dele)))
            print("--------------------")
            '''
            list_last = list_new
        # draw_acc(Acc)
        # draw_num(Add_Num, Del_Num)
        # draw_list_all(Num)
        (pd.DataFrame(data=Add_Num)).to_csv("C:\\Users\\cpz\\Desktop\\threshold_result\\AIR_update_times_"+str(thr)+".csv", header=False, index=False)
        (pd.DataFrame(data=Acc)).to_csv("C:\\Users\\cpz\\Desktop\\threshold_result\\AIR_acc_"+str(thr)+".csv", header=False, index=False)
        print(thr)
    print("OK")