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
model.add(Dense(100, input_dim=2, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 케라스를 위한 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# predict
model.fit(x_data,y_data, batch_size=1,epochs=100)
# evelu

y_pred  = model.predict(x_data)
print(x_data,"의 예측결과 :",y_pred)

result = model.evaluate(x_data,y_data)
# result = model.score(x_data, y_data)
print("model.score_default_acc :", result[1])


# acc = accuracy_score(y_data, y_pred)
# print("accuracy_score : ", acc)
"""
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 : [[0.45214853]
 [0.58779526]
 [0.72539586]
 [0.40686885]]
1/1 [==============================] - 0s 0s/step - loss: 0.4941 - acc: 1.0000
model.score_default_acc : 1.0
"""