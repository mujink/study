# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

# 사이킷 런 데이터셋
# 엘에스티엠으로 모델링
# 덴서와 성능비교
# 다중분류



import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR)
# print(dataset.feature_names)

"""
**Data Set Characteristics:**

    :Number of Instances: 178 (50 in each of three classes)
    :Number of Attributes: 13 numeric, predictive attributes and the class
    :Attribute Information:
                - Alcohol
                - Malic acid
                - Ash
                - Alcalinity of ash
                - Magnesium
                - Total phenols
                - Flavanoids
                - Nonflavanoid phenols
                - Proanthocyanins
                - Color intensity
                - Hue
                - OD280/OD315 of diluted wines
                - Proline

    - class:
            - class_0
            - class_1
            - class_2

    :Summary Statistics:

    ============================= ==== ===== ======= =====
                                   Min   Max   Mean     SD
    ============================= ==== ===== ======= =====
    Alcohol:                      11.0  14.8    13.0   0.8
    Malic Acid:                   0.74  5.80    2.34  1.12
    Ash:                          1.36  3.23    2.36  0.27
    Alcalinity of Ash:            10.6  30.0    19.5   3.3
    Magnesium:                    70.0 162.0    99.7  14.3
    Total Phenols:                0.98  3.88    2.29  0.63
    Flavanoids:                   0.34  5.08    2.03  1.00
    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
    Proanthocyanins:              0.41  3.58    1.59  0.57
    Colour Intensity:              1.3  13.0     5.1   2.3
    Hue:                          0.48  1.71    0.96  0.23
    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
    Proline:                       278  1680     746   315
    ============================= ==== ===== ======= =====
"""
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape) # (178, 13)
print(y.shape) # (178,)

# DNN s


#1.1 Data Preprocessing / train_test_splitm /  MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           #. oneHotEncoder load
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                                #. Set
y = one.transform(y).toarray()      #. transform

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.5, shuffle = True, random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

# y_train = to_categorical(y_train) # one hot encodig for keras.utils
# y_test = to_categorical(y_test) # one hot encodig for keras.utils
# y_val = to_categorical(y_val) # one hot encodig for keras.utils


#2.model
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(Dense(13,activation="relu", input_shape=(13,)))
model.add(Dense(5,activation="relu"))
model.add(Dense(3, activation="softmax"))


#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), batch_size=3, verbose=1, callbacks=[early_stopping])

#4. Evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", accuracy)


y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))
print(y_train[-5:-1])



print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

# print(hist.history['loss'])


# grap
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('wine')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss', 'acc', 'val_acc'])
plt.show()
# Dense model
"""
loss :  0.05921796336770058
acc :  0.9722222089767456

loss :  0.024885298684239388
acc :  0.9722222089767456

loss :  0.004594767466187477
acc :  1.0
"""

# LSTM model
"""
loss :  0.4041748344898224
acc :  0.9444444179534912

loss :  0.23052099347114563
acc :  0.9722222089767456

loss :  0.06399429589509964
acc :  0.9722222089767456
"""