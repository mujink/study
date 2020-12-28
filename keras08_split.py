#. 텐서플로에 케라스에 모델에 시퀀스를 불러옴
from tensorflow.keras.models import Sequential

#. 텐서플로에 케라스에 레이어에 댄스를 불러옴
    #. input_dim => DNN MODEL
    #. xNN 뉴럴 네크워크 (CNN,ANN,VNN,RNN... etc)
from tensorflow.keras.layers import Dense
#. numpy 라이브러리를 np라는 이름으로 불러옴 
import numpy as np
#. numpy에 어레이를 불러옴.
from numpy import array




#1. 데이터
#. @@@@ (트레인 데이터 => 발리데이션 데이터 => 테스트 데이터 => 프레딕트(x에대한 y값을 알고 싶음) 잘 나눠야함 )
x = np.array(range(1,101))
y = np.array(range(101,201))

#. ( 리스트의 슬라이싱 
    #. 원래 데이터에서 트레인 , 발리데이션 , 테스트, 프레딕트를 나눔
x_train = x[:60]   #  0~59 번쨰 까지 :::: 값 1~60
x_val = x[60:80]    #  61~80
x_test = x[80:]     #  81 ~ 100

y_train = y[:60]    #  0~59 번쨰 까지 :::: 값 1~60
y_val = x[60:80]    #  61~80
y_test = x[80:]     #  81 ~ 100
#. ) 리스트의 슬라이싱 




#2. 모델구성
#. @@@@ 순차 모델
model = Sequential()

#. ( 인풋 레이어 아래
    #. @@@@ 설명  DENSE => DNN MODEL
    #. relu => 평가 85% 이상
model.add(Dense(10, input_dim=1, activation='relu'))
#. ) 인풋 레이어 끝

#. ( 히든레이어 
    #. @@@@@@ 노드가 많을 수 록 예측이 정확하고, 연산이 늦어짐.
model.add(Dense(5))
#. ) 히든레이어 끝

#. ( 아웃풋 레이어
model.add(Dense(1))
#. ) 아웃풋 레이어 




#3. 컴파일, 훈련

#. optimizer => 평가 85% 이상
    #. mse, mae => 회귀 평가 지표
    #. accuracy => 분류 평가 지표
    #. batch_size => 한번에 입력하는 데이터 길이
model.compile(loss='mse',optimizer='adam', metrics=['mae'])


#. model.fit(하이퍼 파라미터) 훈련.
#. @@@@ epochs = 훈련횟수 (예측 => 평가 => loss => 수정 ==> 1epochs)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2)


#4. 평가 예측
results = model.evaluate(x_test, y_test, batch_size=20)
print ("mse, mae : ", results)

y_predict = model.predict(x_test)


#. 사이킷 런이 무엇인가
    #. 텐서플로우와 토치와 함께 인공지능 작업 플랫폼 중 하나임


# RMES => np.sqrt => 두 리스트의 제곱 오차 평균
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE :", RMSE(y_test,y_predict))
    # print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

#. R2 평가 지표 (1 = 정확, 0 = 부정확)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )