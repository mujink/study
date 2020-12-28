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

# x_train, x_test, y_train, y_test = train_test_split(입력, 출력값, train_size(x_train, y_train 중 몇% 추출), test_size(x_test, y_test 중 몇% 추출)shuffle(무작위 추출))
    # train_size = 트레인사이즈
    # test_size = 테스트사이즈
""" 
    아래 두 코드는 같은 결과
    x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, train_size = 0.2, shuffle=True)
    x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, test_size = 0.8, shuffle=True)
"""
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                train_size = 0.8, test_size=0.2, shuffle=False)
                                # IF train_size + test_size > 1
"""
    test_size and train_size 의 합이 0~1 범위 안에 있어야함.
    ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
    cross_validation()
"""
                                # ELSE IF train_size + test_size < 1 

"""
(train_size + test_size) - 1 만큼의 데이터가 선택되지 않음. 

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                train_size = 0.7, test_size=0.2, shuffle=False)
    print (x_train) :
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
    49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70]

    print (y_train) :
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
    49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70]

    print (x_test)
    [71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90]
    print (y_test)
    [71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90]
"""
                                # 위 두가지 경우에 대해 확인 후 정리할 것

print (x_train)
print (y_train)
print (x_test)
print (y_test)
# print (x_train.shape)
# print (x_test.shape)
# print (y_train.shape)
# print (y_test.shape)



# print (x_val.shape)
# print (y_val.shape)


# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)
# x_val, x_train, y_val, y_train  = train_test_split(x_train, y_train, train_size = 0.2, shuffle=True)
# print (x_train)
# print (x_train.shape)
# print (x_test.shape)
# print (y_train.shape)
# print (y_test.shape)
# print (x_val.shape)
# print (y_val.shape)



# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_dim=1))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(1))


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train, y_train, epochs=100, validation_data=(x_val,y_val))


# #4. 평가예측
# loss, mse = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('mse : ', mse)

# y_predict = model.predict(x_test)
# # print(y_predict)

# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE :", RMSE(y_test,y_predict))
# # print("mse : ", mean_squared_error(y_test, y_predict))
# print("mse : ", mean_squared_error(y_predict, y_test))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 :",r2 )

# #shuffle = False
# # loss :  1.395869730913546e-05
# # mse :  0.0037330626510083675

# #shuffle = True
# # loss :  0.00032849906710907817
# # mse :  0.015596771612763405

# # Vaildation = 0.2
# # loss :  0.05408924072980881
# # mse :  0.18527519702911377

