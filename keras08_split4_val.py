# 실습 validation_data 를 만들 것!
"""
    train_test_split를 사용할 것!
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터

x = np.array(range(1,101))
y = np.array(range(1,101))

# x_train = x[:60]   #  0~59 번쨰 까지 :::: 값 1~60
# x_val = x[60:80]    #  61~80
# x_test = x[80:]     #  81 ~ 100
 
# y_train = y[:60]    #  0~59 번쨰 까지 :::: 값 1~60
# y_val = x[60:80]    #  61~80
# y_test = x[80:]     #  81 ~ 100


from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(입력, 출력값, train_size(입출력 중 몇%), shuffle(무작위 추출))
    # train_size = 트레인사이즈
    # test_size = 테스트사이즈
""" 
두개 같음
x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, train_size = 0.2, shuffle=True)
x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, test_size = 0.8, shuffle=True)
"""



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)
x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, train_size = 0.2, shuffle=True)
print (x_train)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)
print (x_val.shape)
print (y_val.shape)



#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val,y_val))


#4. 평가예측
loss, mse = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('mse : ', mse)

y_predict = model.predict(x_test)
# print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test,y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )

#shuffle = False
# loss :  1.395869730913546e-05
# mse :  0.0037330626510083675

#shuffle = True
# loss :  0.00032849906710907817
# mse :  0.015596771612763405

# Vaildation = 0.2
# loss :  0.05408924072980881
# mse :  0.18527519702911377

