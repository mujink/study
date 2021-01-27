import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# 평가
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝 모델들
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1 Data Lode
dataset = load_wine()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:5])
print(x.shape, y.shape) # (442, 10) (442,)

print(dataset.feature_names)
print(dataset.DESCR)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)

# x의 값들을 0~1 사이 값으로 줄임 => 가중치를 낮추어 연산 속도가 빨라짐.

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)
# x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1,1)

#2.model
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

model.fit(x_train,y_train)

# 스코어와 프래딕트 평가 예측
y_pred = model.predict(x_test)
print(y_pred)
print(y_train[-5:-1])


result = model.score(x_test,y_test)
print(result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

"""
LinearSVC
0.6944444444444444
accuracy_score :  0.6944444444444444

SVC
0.6388888888888888
accuracy_score :  0.6388888888888888

KNeighborsClassifier
0.6111111111111112
accuracy_score :  0.6111111111111112

DecisionTreeClassifier
0.7222222222222222
accuracy_score :  0.7222222222222222

RandomForestClassifier
1.0
accuracy_score :  1.0

LogisticRegression
0.9722222222222222
accuracy_score :  0.9722222222222222
"""
# Tensorflow
# acc : 1.0