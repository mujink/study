# add..from sklearn.model_selection import KFold, cross_val_score
# preprocessing.. spilt  => KFold , train_test_split
# fit, evl => cross_val_score

# train test 나눈 다음에 train만 발리데이션 하지 말고, 
# kfold 한 후에 train_test_split 사용 ???

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


kfold = KFold(n_splits=5, shuffle=True)

kfold.get_n_splits(x)



# kfold.get_n_splits(x)
# for train_index, test_index in kfold.split(x):
#  print("TRAIN:", train_index, "TEST:", test_index)

#2.model
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]
for model in models :
    print(str(model))
    model = model()
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)
        model.fit( x_train, y_train)
        result = model.score( x_test, y_test)
        print(result)        
        

"""
<class 'sklearn.svm._classes.LinearSVC'>
0.9333333333333333
0.9333333333333333
0.9333333333333333
0.9666666666666667
0.9666666666666667
<class 'sklearn.svm._classes.SVC'>
0.9333333333333333
0.9666666666666667
0.9
0.9333333333333333
0.9666666666666667
<class 'sklearn.neighbors._classification.KNeighborsClassifier'>
0.9666666666666667
0.9666666666666667
0.9666666666666667
0.9666666666666667
0.9333333333333333
<class 'sklearn.tree._classes.DecisionTreeClassifier'>
0.9
0.9333333333333333
0.9333333333333333
0.9666666666666667
0.9333333333333333
<class 'sklearn.ensemble._forest.RandomForestClassifier'>
0.9666666666666667
0.9
0.9666666666666667
0.9333333333333333
0.9666666666666667
<class 'sklearn.linear_model._logistic.LogisticRegression'>
0.9666666666666667
0.9333333333333333
0.9666666666666667
0.8666666666666667
1.0
"""
# Tensorflow
# acc : 1.0