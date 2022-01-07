from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

data = [[1,2,3],[2,3,4],[3,4,5]]
data2 = [1,2,3]
np.random.seed(111)
np.random.shuffle(data)
np.random.shuffle(data2)
print(data)
print(data2)
np.random.seed(111)
np.random.shuffle([1, 2, 3])
print(data2)