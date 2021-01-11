수정필요
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)             # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)               # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],3).astype('float32')/255.

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(1,  input_shape=(1024,3)))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='softmax'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 2000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=2000)

print("loss : ", loss)
print("acc : ", mae)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('cnn fashion_mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss','acc', 'val_acc'])
plt.show()

"""
loss :  2.179011106491089
acc :  0.18860000371932983
R2 : 0.029445838639000067

loss :  2.200120449066162
acc :  0.17419999837875366
R2 : 0.023785003464076047
"""