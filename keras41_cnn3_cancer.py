# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/


import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

# print(datasets.DESCR)           # (569, 30)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)  #(569, 30) , input_dim = 30
# print(y.shape)  #(568, ) # 유방암에 걸렸는지 안 걸렸는지 , output = 1

# print(x[:5])
# print(y)        # 0 or 1 >> classification (이진분류)

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=55)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)
print(y_train.shape)    # (455, 2)
print(y_test.shape)     # (114, 2)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1,1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1],1,1)



#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1,1), input_shape=(30,1,1)))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Conv2D(filters=64, kernel_size=(4,4), padding='same', strides=1))
# model.add(MaxPooling2D(pool_size=3))
model.add(Flatten())
model.add(Dense(30, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# model.summary()

# Compile, Train
model.compile(loss="binary_crossentropy" ,optimizer='adam',metrics=['mae'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
hist = model.fit(x_train, y_train, epochs=500, validation_data=(x_validation, y_validation) ,batch_size=50)

# Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=64)
print("loss : ",loss)
print("mae : ",mae)


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
# print("y_test[:10] :")
# print(y_test[:10])

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
plt.plot(hist.history['val_mae'])

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epochs')
plt.legend(['train loss','val loss','train mae','val mae'])
plt.show()

""""
loss :  0.09642737358808517
mae :  0.027194436639547348
R2 :  0.9998156451838549
"""
