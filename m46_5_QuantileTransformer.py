
# X 통째로 전처리한 놈 => 이러면 안된다 이거야
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

print(np.max(x[0]))

# 1.1 Data Preprocessing
# MinMaxScaler 필수
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
scaler = MinMaxScaler() 
# scaler = StandardScaler() 
# scaler = RobustScaler()

# scaler = QuantileTransformer() # 디폴트 : 균등분포
# scaler = QuantileTransformer(output_distribution="normal") # 정규분포 
scaler.fit(x)
x = scaler.transform(x)

print(np.max(x), np.min(x))
print(np.max(x[0]))


print(np.max(x), np.min(x))
print(np.max(x[0]))
print(np.max(x[1]))
print(np.max(x[2]))
print(np.max(x[3]))
print(np.max(x[4]))
print(np.max(x[5]))
print(np.max(x[6]))
print(np.max(x[7]))
print(np.max(x[8]))
print(np.max(x[9]))
print(np.max(x[10]))
print(np.max(x[11]))
print(np.max(x[12]))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,  test_size=0.2
)


# 2. model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(13,))
d1 = Dense(1000, activation='relu')(input1)
dh = Dense(5, activation='relu')(d1)
dh = Dense(2, activation='relu')(d1)
dh = Dense(3, activation='relu')(d1)
dh = Dense(3, activation='relu')(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, run
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(
    x_train,y_train,
    epochs=200, batch_size=1, validation_data=(x_val, y_val), verbose=2)

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

"""
MinMax
loss :  73.88531494140625
RMSE : 8.59565656220578
RMSE : 73.88531173539128
R2 : -0.03142123212032777

Standard
loss :  12.731224060058594
RMSE : 3.568084115221894
RMSE : 12.731224253298805
R2 : 0.822275162717521
"""