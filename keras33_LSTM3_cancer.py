#  사이킷 런
# LSTM 모델링
#  덴스와 성능 비교
#  이진분류

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
from sklearn.preprocessing import MinMaxScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2.model
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(30,1))
d1 = LSTM(50, activation='sigmoid')(input1)
dh = Dense(50, activation='sigmoid')(d1)
dh = Dense(20, activation='sigmoid')(d1)
dh = Dense(30, activation='sigmoid')(d1)
dh = Dense(30, activation='sigmoid')(dh)
outputs = Dense(1, activation='sigmoid')(dh)

model = Model(inputs =  input1, outputs = outputs)
model.summary()

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), batch_size=3, verbose=1, callbacks=[early_stopping])

#4. Evaluate, predict
loss, acc = model.evaluate(x_test,y_test, batch_size=3)
print("loss : ", loss)
print("acc : ", acc)

y_pred = np.array(model.predict(x_train[-5:-1]))
# print(y_pred.argmax(axis=1))
print(np.where(y_pred>=0.5,1,0))
print(y[-5:-1])

# Dense model
"""
    loss :  0.47400525212287903
    acc :  0.8245614171028137

    loss :  0.32467013597488403
    acc :  0.9736841917037964

    loss :  0.01316804252564907
    acc :  0.9912280440330505

    loss :  0.0458456426858902
    acc :  0.9912280440330505
"""
#  LSTM Model
"""
loss :  0.1545499563217163
acc :  0.9385964870452881

loss :  0.19924435019493103
acc :  0.9473684430122375
"""