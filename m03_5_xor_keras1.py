from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1 data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# model
# model = LinearSVC()
# model = SVC()

model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))

# 케라스를 위한 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# predict
model.fit(x_data,y_data, batch_size=1,epochs=100)
# evelu

y_pred  = model.predict(x_data)
print(x_data,"의 예측결과 :",y_pred)

result = model.evaluate(x_data,y_data)
# result = model.score(x_data, y_data)
print("model.score_default_acc :", result)


# acc = accuracy_score(y_data, y_pred)
# print("accuracy_score : ", acc)
"""
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 : [[0.4746905 ]
 [0.5793483 ]
 [0.65001625]
 [0.73895305]]
model.score_default_acc : [0.7408579587936401, 0.75]

"""