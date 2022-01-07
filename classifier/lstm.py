import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers

# 8k to 8k
dic_size = 8506
sta_size = 50000

x = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_mask24_5w_normal_input.csv", header=None))
y = np.array(pd.read_csv("C:\\Users\\cpz\\Desktop\\traffic_2000w_mask24_5w_output_frequency.csv", header=None))

train_x = x[0:512*dic_size]
train_y = y[0:512*dic_size]
test_x = x[512*dic_size:552*dic_size]
test_y = y[512*dic_size:552*dic_size]
pre_x = x[552*dic_size:590*dic_size]
pre_y = y[552*dic_size:590*dic_size]
train_x = train_x.reshape(512, 1, dic_size)
train_y = train_y.reshape(512, 1, dic_size)
test_x = test_x.reshape(40, 1, dic_size)
test_y = test_y.reshape(40, 1, dic_size)
pre_x = pre_x.reshape(38, 1, dic_size)
pre_y = pre_y.reshape(38, 1, dic_size)

model = Sequential()
model.add(LSTM(16, input_shape=(1, dic_size), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1712, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8506, activation='sigmoid'))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(train_x, train_y, batch_size=16, epochs=10, validation_data=(test_x, test_y), shuffle=False)
model.save('8k_8k_random_model_5w_frequency.h5')
res = (model.predict(pre_x) > 0.5).astype('int64')
accuracy = (pre_y == res).mean()
print(accuracy)

