# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/

import matplotlib.pyplot as plt


import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2]).astype('float32')/255. 
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2])/255. 

print(x_train.shape)    # (60000, 784)
print(x_test.shape)     # (10000, 784)

# print(y_train)
# print(y_test)
# print(y_train.shape)    # (60000, )
# print(y_test.shape)     # (10000, )
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM

model = Sequential()
model.add(LSTM(100, activation="relu", input_shape=(x_train.shape[1],x_train.shape[2])))

model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

model.summary()


# Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
hist = model.fit(x_train, y_train, epochs=9, batch_size=64, validation_data=(x_val, y_val))

# Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss : ",loss)
print("acc : ",acc)


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(np.argmax(y_test[:10],axis=1))

y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('cnn fashion_mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss', 'acc', 'val_acc'])
plt.show()

"""
loss :  0.3269832134246826
acc :  0.8820000290870667
y_test[:10] :
[9 2 1 1 6 1 4 6 5 7]
y_pred[:10] :
[9 2 1 1 6 1 4 6 5 7]
"""