#  argmax 사용  y_predict 최대값 출력
#  sklearn.onehotencoding
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 머신러닝의 분류 모델들
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 분류임 회귀아님


datasets = load_iris()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(datasets.feature_names)
print(datasets.target_names)


print(x.shape)      #(150,4)
print(y.shape)      #(150,3)

#1.1 Data Preprocessing / train_test_splitm /  MinMaxScaler


"""
#  sklearn.onehotencoding

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           #. oneHotEncoder load
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                                #. Set
y = one.transform(y).toarray()      #. transform
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)

# 안쓰는게 더 잘나옴
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)



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
0.9
accuracy_score :  0.9

SVC
0.9666666666666667
accuracy_score :  0.9666666666666667

KNeighborsClassifier
1.0
accuracy_score :  1.0

DecisionTreeClassifier
0.9666666666666667
accuracy_score :  0.9666666666666667

RandomForestClassifier
0.9666666666666667
accuracy_score :  0.9666666666666667

LogisticRegression
0.9666666666666667
accuracy_score :  0.9666666666666667
"""
# Tensorflow
# acc : 1.0