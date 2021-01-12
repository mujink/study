

from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
import matplotlib.pyplot as plt


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],3).astype('float32')/255.
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨

print(x_train.shape)    # (50000, 32, 32, 3)
print(x_test.shape)     # (50000, 32, 32, 3)

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

# 625/625 [==============================] - ETA: 0s - 
# loss: nan - acc: 0.0102WARNING:tensorflow:Model was constructed with shape (None, 3072, 3)
# for input Tensor("lstm_input:0", shape=(None, 3072, 3), dtype=float32),
# but it was called on an input with incompatible shape (None, 1024, 3).
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(3072,3)))
model.add(Dense(50))
model.add(Dense(100, activation='softmax'))

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

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )

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
loss :  4.49699592590332
acc :  0.024900000542402267
R2 : 0.003352384579509097

loss :  4.459139823913574
acc :  0.028200000524520874
R2 : 0.0043950107868227836

loss :  4.435800075531006
acc :  0.032600000500679016
R2 : 0.005046895056631481
"""