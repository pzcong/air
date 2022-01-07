import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report
import csv


def file_write(addr, f):
    file = open(addr, 'w', newline='')
    f_w = csv.writer(file)
    for k in range(500):
        f_w.writerows(f[k])
    file.close()


dic_size = 8506
sta_size = 50000
x = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_mask24_5w_normal_input.csv", header=None))
y = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_2000w_mask24_5w_output_frequency.csv", header=None))
pre_x = x[552*dic_size:590*dic_size]
pre_y = y[552*dic_size:590*dic_size]
pre_x = pre_x.reshape(38, 1, dic_size)
pre_y = pre_y.reshape(38, 1, dic_size)

model = load_model('8k_8k_random_model.h5')
res = (model.predict(pre_x) > 0.5).astype('int64')
np.set_printoptions(threshold=8506)
'''
for i in range(38):
    print(res[i][0][:])

for j in range(500):
    num_1 = 0
    cr = 0
    num_0 = 0
    out_1 = 0
    for i in range(dic_size):
        if pre_y[j][0][i] == 1:
            num_1 += 1
            if res[j][0][i] == 1:
                cr += 1
        if pre_y[j][0][i] == 0 and res[j][0][i] == 1:
            num_0 += 1
        if res[j][0][i] == 1:
            out_1 += 1
    print([num_1, out_1, cr, num_0])
'''