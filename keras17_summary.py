# 외워 머신러닝 딥러닝 기본 프레임워크
import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성 (순차모델 레이어 구성)
from tensorflow.keras.models import Sequential
#. DENSE => DNN MODEL
#. xNN 뉴럴 네크워크 (CNN,ANN,VNN,RNN... etc)
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(9, input_dim=1, activation='linear', name="optimus"))
model.add(Dense(9, activation='linear', name="Prime"))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 실습2 + 과제
# 앙상블 에 대해 서머리를 계산하고 이해한 것을 과제로 제출할 것
"""
    각 인풋레이어를 순회하며 인터페이스에 기록에따라 모델을 컴파일함.
    그리고 머지는 순차적으로 모델을 컴파일하고, 아웃풋이 나누어 질때도 순회하며 모델을 컴파일함.
    파라미터는 같은 시퀀스 모델에서와 같음.
    또, layer의 각 노드와 텐서의 인덱스를 확인할 수 있는 "Connect" Column이 추가됨.
    What does the "[0][0]" of the layers connected to in keras model.summary mean? - Stack Overflow   
"""
# 레이어를 만들때 '네임' 이란 놈에 대해 확인하고 설명할 것
# "네임"을 반드시 써야할 때는 언제인가. 
"""
    "네임"을 입력하면 각 레이어 마다 인덱스를 구분하기에 용이함.
    함수형 앙상블, 분기형 모델을 쓸 때에 "네임" 없이 추적하기 어려움.
    
    함수형 모델링 선언시 머지되는 두 모델의 이름이 같은 경우.
    모델을 선언 할 수 없음. 때문에 머지해야하는 둘 이상의 모델을 선언하기 전에, 둘 이상의 모델의 이름이 겹치지 않아야함.
    참고  :
    https://stackoverflow.com/questions/43452441/keras-all-layer-names-should-be-unique?newreg=0b2bb3dfd8a140eb9b5b263b03777697 
"""

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1 , verbose=0)

#4. 평가, 예측
loss = model.evaluate(x,y,batch_size=1)
print("loss : ", loss)

result = model.predict ([4])
print('result :', result)