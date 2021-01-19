# 45 copy
# 58 2 copy
# 58 3 copy

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)


y_train = x_train
y_tset = x_test



print("=========================================")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("=============")




from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, Reshape
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64, input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Flatten())                                            # 1dim
model.add(Dense(64))
model.add(Dense(784, activation='relu'))
model.add(Reshape((28,28)))
model.add(Dense(1))
model.summary()


#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# ==================================================================================================

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, batch_size= 28)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=28)

print("loss : ", loss)
print("acc : ", mae)

y_predict = model.predict(x_test)
