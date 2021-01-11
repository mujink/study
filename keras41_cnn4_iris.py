# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/


import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()
x = datasets.data
y = datasets.target
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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1,1)


print(y_train.shape)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 1, kernel_size=(1,1), input_shape=(x_train.shape[1],1,1)))

model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=500, verbose=1, batch_size= 6, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=6)

print("loss : ", loss)
print("acc : ", acc)

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
plt.title('cnn iris')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'acc'])
plt.show()

"""
loss :  0.05478470399975777
acc :  0.9666666388511658
R2 : 0.9287338949035523
"""