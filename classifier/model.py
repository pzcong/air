from keras.layers import Dense
from keras.models import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
Y = iris.target
x = X[Y < 2, :2]
print(x.shape)
y = Y[Y < 2]
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=6)
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=2))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
