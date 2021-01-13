# 실습
# conv1d로 코딩하시오

import numpy as np
x_train = '../data/npy/mnist_x_train.npy'
x_test = '../data/npy/mnist_x_test.npy'
y_train = '../data/npy/mnist_y_train.npy'
y_test = '../data/npy/mnist_y_test.npy'
x_train = np.load(x_train)
x_test = np.load(x_test)
y_train = np.load(y_train)
y_test = np.load(y_test)

from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]).astype('float32')/255.

#2. Modling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, Conv1D, Flatten,MaxPool1D

model = Sequential()
model.add(Conv1D(10,2 ,input_shape=(x_train.shape[1],x_train.shape[2]) ,padding='same'))
model.add(MaxPool1D(pool_size=(2)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# model = Sequential()
# model.add(LSTM(100, activation="relu", input_shape=(x_train.shape[1],x_train.shape[2])))
# model.add(Dense(50))
# model.add(Dense(10, activation='softmax'))
# model.summary()


# Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
model.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.3)

# Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
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
 
""" dense
loss :  0.058254096657037735
acc :  0.9825000166893005
y_test[:10] :
[7 2 1 0 4 1 4 9 5 9]
y_pred[:10] :
[7 2 1 0 4 1 4 9 6 9]
"""

"""c1d
loss :  0.1448270082473755
acc :  0.9713000059127808
y_test[:10] :
[7 2 1 0 4 1 4 9 5 9]
y_pred[:10] :
[7 2 1 0 4 1 4 9 6 9]
"""