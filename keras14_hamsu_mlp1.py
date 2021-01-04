# 다:1 mlp
# keras10_mlp2.py 를 함수형으로 바꾸시오.


import numpy as np

#1 data
x = np.array([range(100), range(301,401), range(1,101)])
y = np.array(range(711,811))
print(x.shape)          #(3,100)
print(y.shape)          #(100,)
 
x = np.transpose(x)
# print(x)
print(x.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
#1.2 pred2
x_pred2 = np.array([[0,301,1]])
#2. modelconfig
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# from keras.layers import Dense

input1 = Input(shape=(3,))
Hlayer1 = Dense(5, activation='relu')(input1)
Hlayer2 = Dense(3)(Hlayer1)
Hlayer3 = Dense(4)(Hlayer2)
outputs = Dense(1)(Hlayer3)
model = Model(inputs =  input1, outputs = outputs)
model.summary()

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_split=0.2)

#4 Evaluation validation
loss, mae = model.evaluate(x_train,y_train)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test,y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )
y_Pred2 = model.predict(x_pred2)
print("y_pred2 : " , y_Pred2)