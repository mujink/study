#  실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
#  총 6개의 파일을 완성하시오.

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

#1.1 Data Preprocessing

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=3
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


# 2. model

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input


# model = Sequential()
# model.add(Dense(128, input_shape=(10,), activation='linear'))
# model.add(Dense(64, activation='linear'))
# model.add(Dense(64, activation='linear'))
# model.add(Dense(64, activation='linear'))
# model.add(Dense(32, activation='linear'))
# model.add(Dense(32, activation='linear'))
# model.add(Dense(32, activation='linear'))
# model.add(Dense(1))

input1 = Input(shape=(10,))
d1 = Dense(1000, activation='linear')(input1)
dh = Dense(3)(d1)
dh = Dense(3)(d1)
dh = Dense(3)(d1)
dh = Dense(100)(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, Train

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(
    x_train,y_train,
    epochs=5000, batch_size= 6, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping]
    # epochs=5000, batch_size= 10, validation_data=(x_val, y_val), verbose=2
)

# 4 Evaluation validation
loss, mae= model.evaluate(x_test, y_test, batch_size=6)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict) )



from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )

"""
Case 1 : keras18_EarlyStopping

loss :  5351.4169921875
mae :  55.89436721801758
RMSE : 73.15337838274678

Case 2 : activation='liner'

loss :  3436.531494140625
mae :  46.62599182128906
RMSE : 58.62193665989579
R2 : 0.4103986022877909

loss :  2843.052001953125
mae :  43.638343811035156
RMSE : 53.320280511147125
R2 : 0.5122210756446386

Case 3 : Non callbacks , Epochs = 5000

loss :  2812.398193359375
mae :  42.80008316040039
RMSE : 53.032047707759
R2 : 0.5174803834393196

Case 4 : callbacks = 100 , activation='liner' , Epochs = 5000

loss :  3266.852783203125
mae :  44.29716110229492
RMSE : 57.15638481328284
R2 : 0.3966695113417922

loss :  3221.448486328125
mae :  44.62709426879883
RMSE : 56.75780512789122
R2 : 0.4050548142518875
"""