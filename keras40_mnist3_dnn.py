# 주말과제
# dence 모델로 구성 input_shape=(28*28,)

#  인공지능게의  helloe world => Mnist!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]).astype('float32')/255.
# (x_test.reshap(10000, 28, 28, 1))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(784,)))

model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. Compile, train / binary_corssentropy

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=5, verbose=1, batch_size= 30)

#4. Evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", accuracy)



y_pred = model.predict(x_test[:10])
print(y_pred.argmax(axis=1))
print(y_test.argmax(axis=1))

print("    예상 출력     /    실제  값     ")
# print("    ",y_pred[:10],"     /    ", y_test[:10])

""" Dnn(28*28무척 빠름)
loss :  0.08407819271087646
acc :  0.9757000207901001
[7 2 1 0 4 1 4 9 6 9]
[7 2 1 ... 4 5 6]
"""
