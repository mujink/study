# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1 Data Lode
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(x[:10])
print(x.shape, y.shape) # (442, 10) (442,)

print(dataset.feature_names)
print(dataset.DESCR)


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

# model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(10, (2,2), padding='valid'))
# model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(100))
model.add(Dense(1, activation='linear'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=500, verbose=1, batch_size= 10, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['mae'])
plt.title('cnn boston')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'mae'])
plt.show()


"""
loss :  2983.510986328125
mae :  43.99277114868164
R2 : 0.48812279363257227
"""