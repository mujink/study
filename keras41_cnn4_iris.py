# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/


import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data 
y = dataset.target 
print(datasets.DESCR)
print(datasets.feature_names)
print(datasets.target_names)

"""
 :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988
"""
print(x.shape)      #(150,4)
print(y.shape)      #(150,3)

#1.1 Data Preprocessing / train_test_splitm /  MinMaxScaler
# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# 다중 분류일 때, y값 전처리 One hot Encoding (1) tensorflow.keras 사용
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical # 옛날 버전

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)
# print(y_train)
# print(y_test)
print(y_train.shape)    # (120, 3) >>> output = 3
print(y_test.shape)     # (30, 3)

# 다중 분류일 때, y값 전처리 One hot Encoding (2) sklearn 사용 >>> keras22_1_iris1_(2)skelarn.py


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1,1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1],1,1)


#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1,1), input_shape=(4,1,1)))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Conv2D(filters=64, kernel_size=(4,4), padding='same', strides=1))
# model.add(MaxPooling2D(pool_size=3))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
hist = model.fit(x_train, y_train, epochs=500, validation_data=(x_validation, y_validation), batch_size=10)

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
r2 = r2_score(y_test[:10], y_predict)
print("R2 : ", r2)

import matplotlib.pyplot as plt 

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss','val loss','train acc','val acc'])
plt.show()

"""
loss :  0.07838401943445206
acc :  0.9666666388511658
y_test[:10] :
[1 1 1 0 1 1 0 0 0 2]
y_pred[:10] :
[1 1 1 0 1 1 0 0 0 2]
R2 :  0.9982918840625906
"""
