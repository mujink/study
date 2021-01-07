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

# 1.1 Data Preprocessing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)


# 2. model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(13,))
d1 = Dense(13, activation='relu')(input1)
dh = Dense(5, activation='relu')(d1)
dh = Dense(7, activation='relu')(dh)
dh = Dense(8, activation='relu')(dh)
dh = Dense(9, activation='relu')(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, run
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(
    x_train,y_train,
    epochs=1000, batch_size=13, validation_split=0.2, verbose=2, callbacks=[early_stopping])
 
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

# graph

# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('wine')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss', 'mae', 'val_mae'])
plt.show()