#  사이킷 런
# LSTM 모델링
#  덴스와 성능 비교
# 다중분류

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

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

# y_train = to_categorical(y_train) # one hot encodig for keras.utils
# y_test = to_categorical(y_test) # one hot encodig for keras.utils
# y_val = to_categorical(y_val) # one hot encodig for keras.utils


#2.model
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(50,activation="relu", input_shape=(4,1)))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(3, activation="softmax"))


# input1 = Input(shape=(4,))
# d1 = Dense(1000, activation='sigmoid')(input1)
# dh = Dense(50, activation='sigmoid')(d1)
# dh = Dense(20, activation='sigmoid')(d1)
# dh = Dense(30, activation='sigmoid')(d1)
# dh = Dense(30, activation='sigmoid')(dh)
# outputs = Dense(1, activation='softmax')(dh)

# model = Model(inputs =  input1, outputs = outputs)
# model.summary()

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), batch_size=3, verbose=1, callbacks=[early_stopping])

#4. Evaluate, predict
loss, accuracy, mae = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", accuracy)


y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))
print(y_train[-5:-1])

# Dense model
"""
loss :  0.039841461926698685
acc :  0.9666666388511658
"""
# LSTM model
"""
loss :  0.068038210272789
acc :  0.9666666388511658

loss :  0.08814556151628494
acc :  0.9666666388511658

loss :  0.08931181579828262
acc :  0.9666666388511658
"""