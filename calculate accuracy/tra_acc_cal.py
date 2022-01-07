import pandas as pd
import numpy as np
import random

sta_size = 100000
rule_size = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000, 200000, 300000, 500000]
for k in range(20):
    thr = k+1
    traffic = pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_1760w_mask24_rule.csv", header=None).iloc[:].values
    traffic = np.squeeze(traffic, axis=1)
    traffic = list(traffic)
    rule_set = random.sample(range(1, 600000), rule_size[k])
    Num = []
    Acc = []
    print(k)
    for i in range(5):
        tmp = traffic[i * sta_size:(i + 1) * sta_size]
        hit = 0
        exc_time = 0
        for j in tmp:
            if j in rule_set:
                hit += 1
                rule_set.remove(j)
            else:
                exc_time += 1
                del (rule_set[0])  # delete the LRU element
            rule_set.append(j)  # put the last access to the last of list
        print(exc_time)
        print(hit/sta_size)
        Num.append(exc_time)
        Acc.append(hit / sta_size)
    (pd.DataFrame(data=Num)).to_csv("C:\\Users\\cpz\\Desktop\\threshold_result2\\tra_update_times_LRU_" + str(thr) + ".csv", header=False, index=False)
    (pd.DataFrame(data=Acc)).to_csv("C:\\Users\\cpz\\Desktop\\threshold_result2\\tra_acc_LRU_" + str(thr) + ".csv", header=False, index=False)
print("OK")