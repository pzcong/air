import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

tra_real = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:].values
print(tra_real.shape)
tra_real = pd.read_csv("C:\\Users\\cpz\\Desktop\\1259-1302_10_input_sorted.csv", header=None).iloc[:, 1124:1300].values
print(tra_real.shape)
