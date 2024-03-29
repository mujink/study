# minmaxsclar
#1. Data
import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,)
print('====================')
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))     # 711.0 0.0
print(dataset.feature_names)
# 1.1 Data Preprocessing
x = x / 711.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)


# 2. model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(13,))
d1 = Dense(13, activation='relu')(input1)
dh = Dense(11, activation='relu')(d1)
dh = Dense(9, activation='relu')(dh)
dh = Dense(7, activation='relu')(dh)
# dh = Dense(5, activation='relu')(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, run
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(
    x_train,y_train,
    epochs=1400, batch_size=13, validation_split=0.2, verbose=2)

# 4 Evaluation validation
loss, mae= model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict) )

# mse... 
MSE1 = mean_squared_error(y_test, y_predict)

print("RMSE :", MSE1 )

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )