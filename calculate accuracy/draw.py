import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# 替换次数


def visualize_acc(x1, x2):
    color_blue = ['midnightblue', 'blue', 'dodgerblue', 'deepskyblue', 'cornflowerblue', 'c', 'steelblue', 'lightskyblue', 'teal', 'aqua']
    color_red = ['lightcoral', 'deeppink', 'magenta', 'crimson', 'darkorange', 'maroon', 'red', 'chocolate', 'goldenrod', 'gold']
    for ind in range(len(x1)):
        plt.plot(x1[ind], color=color_red[ind])
        plt.plot(x2[ind], color=color_blue[ind])
    plt.plot([], linestyle='-', color='r', label='Red Series: AIR')
    plt.plot([], linestyle='-', color='b', label='Blue Series: Traditional')
    plt.xlabel(xlabel='Period')
    plt.ylabel(ylabel='Hit rate')
    plt.xlim([0, 180])  # x轴边界
    plt.xticks(range(0, 181, 20))  # 设置x刻度
    plt.ylim([0.7, 1])  # y轴边界
    # plt.yticks(range(0, 15000, 1000))  # 设置y刻度
    # plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='x')
    plt.legend()
    plt.show()


def visualize_update_times(x1, x2):
    color_blue = ['midnightblue', 'blue', 'dodgerblue', 'deepskyblue', 'cornflowerblue', 'c', 'steelblue',
                  'lightskyblue', 'teal', 'aqua']
    color_red = ['lightcoral', 'deeppink', 'magenta', 'crimson', 'darkorange', 'maroon', 'red', 'chocolate',
                 'goldenrod', 'gold']
    for ind in range(len(x1)):
        plt.plot(np.log10(x1[ind]), color=color_red[ind])
        plt.plot(np.log10(x2[ind]), color=color_blue[ind])
    plt.plot([], linestyle='-', color='r', label='Red Series: AIR')
    plt.plot([], linestyle='-', color='b', label='Blue Series: Traditional')
    plt.xlabel(xlabel='Period')
    plt.ylabel(ylabel='Replacement times(log)')
    plt.xlim([0, 180])  # x轴边界
    plt.xticks(range(0, 181, 20))  # 设置x刻度
    plt.ylim([1, 5])  # y轴边界
    # plt.yticks(range(0, 1401, 200))  # 设置y刻度
    plt.legend()
    plt.show()


def draw_threshold_cap(valu):
    x1 = np.arange(1, 21)
    plt.bar(x1, valu, width=0.9, color='darkorange', align='center', alpha=1)
    plt.xlabel(xlabel='Threshold ')
    plt.ylabel(ylabel='Average TCAM size')
    plt.xlim([1, 21])
    plt.xticks(range(0, 21, 1))
    plt.ylim([0, 3000])
    #for a, b in zip(x1, valu):
        #plt.text(a, b+230, '%.0f' % b, ha='center', va='top', fontsize=10, rotation=30)
    # plt.legend()
    plt.show()


def draw_threshold_acc(valu):
    x1 = np.arange(1, 21)
    plt.bar(x1, valu, width=0.9, color='darkolivegreen', align='center', alpha=1)
    plt.xlabel(xlabel='Threshold ')
    plt.ylabel(ylabel='Average hit rate')
    plt.xlim([1, 21])
    plt.xticks(range(0, 21, 1))
    plt.ylim([0.5, 1])
    for a, b in zip(x1, valu):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='top', fontsize=10, rotation=40)
    # plt.legend()
    plt.show()


def filter_times(valu):
    # plt.axhline(y=8598, color='red', ls='-.', lw=1, alpha=1, label="Original times of prediction (8598)")
    for k in range(len(valu)):
        valu[k] *= 8598
    x1 = np.arange(4, 17)
    plt.bar(x1, valu, width=0.9, color='cornflowerblue', align='center', alpha=1)
    plt.xlabel(xlabel='Number of history periods')
    plt.ylabel(ylabel='Times of Prediction')
    plt.xlim([3, 17])
    plt.xticks(range(3, 17, 1))
    plt.ylim([0, 10000])
    for a, b in zip(x1, valu):
        plt.text(a, b + 350, '%.0f' % b, ha='center', va='top', fontsize=10, rotation=0)
    # plt.legend()
    plt.show()


def filter_acc(valu):
    x1 = np.arange(4, 17)
    plt.axhline(y=0.847596, color='red', ls='-.', lw=1, alpha=1, label="Original hit rate(0.8475)")
    plt.bar(x1, valu, width=0.9, color='olivedrab', align='center', alpha=1)
    plt.xlabel(xlabel='Number of history periods ')
    plt.ylabel(ylabel='Hit rate')
    plt.xlim([3, 17])
    plt.xticks(range(3, 17, 1))
    plt.ylim([0.8, 0.9])
    for a, b in zip(x1, valu):
        plt.text(a, b+0.005, '%.5f' % b, ha='center', rotation=30, va='center', fontsize=10)
    plt.legend()
    plt.show()


def draw_2vol_acc(list1, list2, list3):  # 可视化增减数量
    x1 = np.arange(1, 11)
    plt.bar(x1 - 0.45, list1, width=0.3, color="white", edgecolor='tomato', align='edge', label='AIR', alpha=1, hatch='//')
    plt.bar(x1 - 0.15, list2, width=0.3, color="white", edgecolor='dodgerblue', align='edge', label='Traditional', alpha=1, hatch='-')
    plt.bar(x1 + 0.15, list3, width=0.3, color="white", edgecolor='limegreen', align='edge', label='LRU', alpha=1, hatch='.')
    plt.xlabel(xlabel='Threshold', size=18)
    plt.ylabel(ylabel='Hit rate', size=18)
    plt.xticks(range(1, 11, 1), size=15)
    plt.yticks(size=15)
    plt.ylim([0.0, 1.3])
    plt.legend(fontsize=15)
    plt.show()
    """
    for a, b in zip(x1, list2):
        plt.text(a-0.2, b+0.005, '%.3f' % b, ha='center', rotation=20, va='center', fontsize=10)
    for a, b in zip(x1, list3):
        plt.text(a+0.3, b+0.005, '%.3f' % b, ha='center', rotation=20, va='center', fontsize=10)
    """


def draw_2vol_times(list1, list2, list3):  # 可视化增减数量
    list1 = np.log10(list1)
    list2 = np.log10(list2)
    list3 = np.log10(list3)
    x1 = np.arange(1, 11)
    plt.bar(x1-0.45, list1, width=0.3, color="white", edgecolor='tomato', align='edge', label='AIR', alpha=1, hatch='//')
    plt.bar(x1-0.15, list2, width=0.3, color="white", edgecolor='dodgerblue', align='edge', label='Traditional', alpha=1, hatch='-')
    plt.bar(x1+0.15, list3, width=0.3, color="white", edgecolor='limegreen', align='edge', label='LRU', alpha=1, hatch='.')
    plt.xlabel(xlabel='Threshold', size=18)
    plt.ylabel(ylabel='Replacement times(log)', size=18)
    plt.xticks(range(1, 11, 1), size=15)
    plt.yticks(size=15)
    plt.ylim([0, 6])
    plt.legend(fontsize=15)
    plt.show()
'''
    for a, b in zip(x1, list2):
        plt.text(a-0.2, b+0.05, '%.2f' % b, ha='center', rotation=0, va='center', fontsize=10)
    for a, b in zip(x1, list3):
        plt.text(a+0.3, b+0.05, '%.2f' % b, ha='center', rotation=0, va='center', fontsize=10)
'''


def draw_2_times(x1, x2):
    x1 = np.log10(x1)
    x2 = np.log10(x2)
    plt.plot(x1, linestyle='-', color='tomato', label='AIR')
    plt.plot(x2, linestyle='-', color='dodgerblue', label='Traditional')
    plt.xlabel(xlabel='Threshold')
    plt.ylabel(ylabel='Average times of replacement')
    plt.xlim([0, 21])  # x轴边界
    plt.xticks(range(0, 21))  # 设置x刻度
    plt.ylim([1, 5])  # y轴边界
    # plt.yticks(range(0, 1401, 200))  # 设置y刻度
    plt.legend()
    plt.show()


def filter_both(x1, x2):
    x = np.arange(4, 17)
    fig, ax = plt.subplots()
    ax_sub = ax.twinx()

    ax.bar(x,  x1, width=0.9, color='dodgerblue', align='center', alpha=1, label="Times of prediction")
    ax_sub.plot(x, x2, label="Hit rate with Filter", color='lime', lw=1)
    ax_sub.axhline(y=0.84745, color='red', ls='-.', lw=1, alpha=0.8, label="Original hit rate")
    ax.set_xlim(3, 17)
    ax.set_xticks(np.arange(3, 17, 1))
    ax.tick_params(labelsize=12)
    ax_sub.tick_params(labelsize=12)
    ax.set_xlabel("History length", size=15)
    ax.set_ylim(0, 10000)
    ax_sub.set_ylim(0.8470, 0.8480)
    ax.set_ylabel("Times of prediction", size=15)
    ax_sub.set_ylabel("Hit rate", size=15)
    '''
    for a, b in zip(x, x1):
        ax.text(a, b + 350, '%.0f' % b, ha='center', va='top', fontsize=9, rotation=0)
    '''
    plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='x')
    plt.legend(loc=0, fontsize=12)
    fig.show()


if __name__ == '__main__':
    """
    times_tra = pd.read_csv("C:\\Users\\cpz\\Desktop\\tra_update_times.csv", header=None).iloc[:].values
    times_AIR = pd.read_csv("C:\\Users\\cpz\\Desktop\\AIR_update_times.csv", header=None).iloc[:].values
    visualize_update_times(times_AIR, times_tra)
    
    acc_tra = pd.read_csv("C:\\Users\\cpz\\Desktop\\tra_acc.csv", header=None).iloc[:].values
    acc_AIR = pd.read_csv("C:\\Users\\cpz\\Desktop\\AIR_acc.csv", header=None).iloc[:].values
    visualize_acc(acc_AIR, acc_tra)

    val = pd.read_csv("C:\\Users\\cpz\\Desktop\\threshold.csv", header=None).iloc[:].values
    thre_cap = val[:, 0]
    thre_acc = val[:, 1]
    # draw_threshold_acc(thre_acc)
    draw_threshold_cap(thre_cap)
"""
    val = pd.read_csv("C:\\Users\\cpz\\Desktop\\filter.csv", header=None).iloc[:].values
    fil_times = val[:, 0]
    for k in range(len(fil_times)):
        fil_times[k] *= 8598
    fil_acc = val[:, 2]
    filter_both(fil_times, fil_acc)
    # filter_times(fil_times)
    # filter_acc(fil_acc)
"""    
    tra = []
    air = []
    for i in [1, 5, 10, 20]:
        tra.append(pd.read_csv("C:\\Users\\cpz\\Desktop\\threshold_result\\tra_update_times_"+str(i)+".csv", header=None).iloc[:].values)
        air.append(pd.read_csv("C:\\Users\\cpz\\Desktop\\threshold_result\\AIR_update_times_" + str(i) + ".csv", header=None).iloc[:].values)
    visualize_update_times(air, tra)

    tmp = pd.read_csv("C:\\Users\\cpz\\Desktop\\threshold.csv", header=None).iloc[:].values
    thr_air_acc = tmp[0:10, 4]
    thr_tra_acc = tmp[0:10, 5]
    thr_lru_acc = tmp[0:10, 6]
    thr_air_times = tmp[0:10, 1]
    thr_tra_times = tmp[0:10, 2]
    thr_lru_times = tmp[0:10, 3]
    draw_2vol_acc(thr_air_acc, thr_tra_acc, thr_lru_acc)
    draw_2vol_times(thr_air_times, thr_tra_times, thr_lru_times)
"""