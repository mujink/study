import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# 평가
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝 모델들
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1 Data Lode
dataset = load_diabetes()
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

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2.model
model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

model.fit(x_train,y_train)

# 스코어와 프래딕트 평가 예측
y_pred = model.predict(x_test)
print(y_pred)
print(y_train[-5:-1])

result = model.score(x_test,y_test)
print(result)

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)

"""
LinearRegression
0.4975430147899931
r2_score :  0.4975430147899931

KNeighborsRegressor
0.4132998100289288
r2_score :  0.4132998100289288

DecisionTreeRegressor
0.06719614213126723
r2_score :  0.06719614213126723

RandomForestRegressor
0.4810450228861576
r2_score :  0.4810450228861576
"""
# Tensorflow
# r2_score : 0.5174803834393196