#  사이킷 런
# LSTM 모델링
#  덴스와 성능 비교
# 회귀



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
from tensorflow.keras.layers import Dense, Input, LSTM


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
d1 = LSTM(1000, activation='linear')(input1)
dh = Dense(50, activation='relu')(d1)
dh = Dense(50, activation='relu')(d1)
dh = Dense(50, activation='relu')(d1)
dh = Dense(100, activation='linear')(dh)
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
    epochs=5000, batch_size= 6, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping]
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


# Dense model
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
# LSTM model
"""
loss :  3413.84912109375
mae :  47.19691467285156
RMSE : 58.42814789753751
R2 : 0.36952189484442377

loss :  3039.118408203125
mae :  42.75789260864258
RMSE : 55.128199003154684
R2 : 0.43872799810290153

loss :  3205.9423828125
mae :  44.91044235229492
RMSE : 56.62103730054887
R2 : 0.40791860799545976

loss :  3147.7890625
mae :  44.5740852355957
RMSE : 56.105158591955394
R2 : 0.4186584582188735

loss :  3241.644287109375
mae :  44.615570068359375
RMSE : 56.93543673690737
R2 : 0.40132505612120084
"""
