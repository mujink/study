# add..from sklearn.model_selection import KFold, cross_val_score
# preprocessing.. spilt  => KFold
# fit, evl => cross_val_score

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 머신러닝의 분류 모델들
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 분류임 회귀아님

import warnings

warnings.filterwarnings('ignore')

datasets = load_iris()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
# print(datasets.feature_names)
# print(datasets.target_names)


print(x.shape)      #(150,4)
print(y.shape)      #(150,3)

#1.1 Data Preprocessing / KFold

# data splits
kfold = KFold(n_splits=5, shuffle=True)
# kfold.get_n_splits(x)
# for train_index, test_index in kfold.split(x):
#  print("TRAIN:", train_index, "TEST:", test_index)
"""
#  sklearn.onehotencoding

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           #. oneHotEncoder load
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                                #. Set
y = one.transform(y).toarray()      #. transform
# """

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)


# #2.model
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = LogisticRegression()

scores = cross_val_score(model, x, y, cv= kfold)
print('scores : ', scores)


"""
model.fit(x_train,y_train)


# 스코어와 프래딕트 평가 예측
y_pred = model.predict(x_test)
print(y_pred)
print(y_train[-5:-1])


result = model.score(x_test,y_test)
print(result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

LinearSVC
0.9
accuracy_score :  0.9
scores :  [0.86666667 1.         1.         0.96666667 1.        ]

SVC
0.9666666666666667
accuracy_score :  0.9666666666666667
scores :  [0.96666667 0.9        0.96666667 1.         0.96666667]

KNeighborsClassifier
1.0
accuracy_score :  1.0
scores :  [0.96666667 0.96666667 0.96666667 0.93333333 1.        ]

DecisionTreeClassifier
0.9666666666666667
accuracy_score :  0.9666666666666667
scores :  [0.96666667 0.93333333 0.96666667 0.96666667 0.86666667]

RandomForestClassifier
0.9666666666666667
accuracy_score :  0.9666666666666667
scores :  [0.93333333 0.96666667 1.         0.93333333 0.93333333]

LogisticRegression
0.9666666666666667
accuracy_score :  0.9666666666666667
scores :  [1.         1.         0.96666667 0.96666667 0.86666667]
"""
# Tensorflow
# acc : 1.0