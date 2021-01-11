# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/


import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape)      #(569,30)
print(y.shape)      #(569,)
print(x[:5])
print(y)

print(datasets.target_names)
print(datasets.feature_names)
# print(datasets.DESCR)

"""Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

    :Summary Statistics:

    ===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097

    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03

    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
    ===================================== ====== ======

"""

#1.1 Data Preprocessing / train_test_splitm /  MinMaxScaler


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,  test_size=0.2
)

# x의 값들을 0~1 사이 값으로 줄임 => 가중치를 낮추어 연산 속도가 빨라짐.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1,1)
# (x_test.reshap(506, 13, 1, 1))


from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 1, kernel_size=(1,1), input_shape=(x_train.shape[1],1,1)))

model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=500, verbose=1, batch_size= 10, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("mae : ", acc)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('cnn cancer')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'acc'])
plt.show()

"""
loss :  0.12232113629579544
mae :  0.9736841917037964
R2 : 0.9153961797099273
"""