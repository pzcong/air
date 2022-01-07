import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

dic_size = 8506
sta_size = 10000

x = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_2000w_ipv4_mask24_random_rule.csv"))
y = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_mask24_1w_random_output_intersection_1w21w.csv"))

train_x = x[0:2048*sta_size]
train_y = y[0:2048*sta_size]
test_x = x[2048*sta_size:2304*sta_size]
test_y = y[2048*sta_size:2304*sta_size]
pre_x = x[2304*sta_size:2404*sta_size]
pre_y = y[2304*sta_size:2404*sta_size]
train_x = train_x.reshape(2048, 1, sta_size)
train_y = train_y.reshape(2048, 1, sta_size)
test_x = test_x.reshape(256, 1, sta_size)
test_y = test_y.reshape(256, 1, sta_size)
pre_x = pre_x.reshape(100, 1, sta_size)
pre_y = pre_y.reshape(100, 1, sta_size)
model = Sequential()
model.add(LSTM(32, input_shape=(1, sta_size), return_sequences=True))
model.add(Dense(20000, activation='relu'))
model.add(Dense(10000, activation='sigmoid'))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=16, epochs=5, validation_data=(test_x, test_y), shuffle=False)
model.save('my_model.h5')
res = (model.predict(pre_x) > 0.5).astype('int64')
accuracy = (pre_y == res).mean()
print(accuracy)
