#  다차원 댄스 모델?
#  (n, 32, 32, 3) => (n,32, 32, 3)


from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)             # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)               # (10000, 32, 32, 3) (10000, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

y_train = x_train
y_test = x_test

print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, Reshape
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(30, input_shape=(32,32,3),activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()


#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=1, verbose=1, batch_size= 64)

#4. Evaluate, predict
print(x_test.shape)
print(y_test.shape)
loss, mae = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

