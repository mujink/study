수정필요

# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오

#  인공지능게의  helloe world => Mnist!!
import numpy as np

from sklearn.datasets import load_boston #보스턴 집값에 대한 데이터 셋을 교육용으로 제공하고 있다.

dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target # target : x와 y 가 분리한다.

# 다 : 1 mlp 모델을 구성하시오

print(x.shape)  # (506, 13) input = 13
print(y.shape)  # (506, )   output = 1
print('==========================================')

# ********* 데이터 전처리 ( MinMax ) *********


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0      ----> 최댓값 1.0 , 최솟값 0.0
print(np.max(x[0]))         # max = 396.9

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1,1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1],1,1)


#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(13,1,1)))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Conv2D(filters=64, kernel_size=(4,4), padding='same', strides=1))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

# model.summary()

# Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_validation, y_validation) ,batch_size=8)

# Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ",loss)
print("mae : ",mae)


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(y_test[:10])

# print(np.argmax(y_test[:10],axis=1))

y_predict = model.predict(x_test[:10])
# print("y_pred[:10] :")
# print(y_predict[:10])


from sklearn.metrics import r2_score
r2 = r2_score(y_test[:10], y_predict)
print("R2 : ", r2)

import matplotlib.pyplot as plt 

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['mae'])

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epochs')
plt.legend(['train loss','val loss','train mae','val mae'])
plt.show()

# loss :  9.123716354370117
# mae :  2.2711946964263916
# y_test[:10] :
# [16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1]
# R2 :  0.946834550299679
