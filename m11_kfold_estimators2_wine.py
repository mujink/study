# m09 => Fold, cross_val_score

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        cores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)

        print(name, '의 정답률 : ', cores)        
        # print(name, '의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        # continue
        print(name, "은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
import sklearn
print(sklearn.__version__) # 0.23.2

"""
AdaBoostClassifier 의 정답률 :  [0.93103448 0.86206897 0.53571429 0.82142857 0.96428571]
BaggingClassifier 의 정답률 :  [0.96551724 0.89655172 0.96428571 0.92857143 1.        ]
BernoulliNB 의 정답률 :  [0.44827586 0.37931034 0.25       0.46428571 0.5       ]
CalibratedClassifierCV 의 정답률 :  [0.93103448 0.96551724 0.82142857 0.82142857 0.96428571]
CategoricalNB 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ComplementNB 의 정답률 :  [0.62068966 0.79310345 0.67857143 0.64285714 0.53571429]
DecisionTreeClassifier 의 정답률 :  [0.93103448 0.82758621 0.92857143 0.89285714 0.92857143]
DummyClassifier 의 정답률 :  [0.48275862 0.37931034 0.32142857 0.25       0.25      ]
ExtraTreeClassifier 의 정답률 :  [0.86206897 0.89655172 0.82142857 0.82142857 0.89285714]
ExtraTreesClassifier 의 정답률 :  [1.         0.96551724 1.         1.         1.        ]
GaussianNB 의 정답률 :  [1.         0.96551724 1.         0.96428571 0.92857143]
GaussianProcessClassifier 의 정답률 :  [0.48275862 0.55172414 0.35714286 0.42857143 0.35714286]
GradientBoostingClassifier 의 정답률 :  [0.93103448 0.93103448 0.96428571 1.         0.96428571]
HistGradientBoostingClassifier 의 정답률 :  [0.96551724 0.96551724 0.92857143 0.96428571 1.        ]
KNeighborsClassifier 의 정답률 :  [0.75862069 0.72413793 0.71428571 0.67857143 0.75      ]
LabelPropagation 의 정답률 :  [0.55172414 0.4137931  0.46428571 0.35714286 0.5       ]
LabelSpreading 의 정답률 :  [0.37931034 0.62068966 0.32142857 0.42857143 0.46428571]
LinearDiscriminantAnalysis 의 정답률 :  [1.         1.         0.96428571 0.96428571 1.        ]
LinearSVC 의 정답률 :  [0.72413793 0.96551724 0.82142857 0.85714286 0.92857143]
LogisticRegression 의 정답률 :  [0.93103448 0.96551724 0.96428571 0.89285714 0.96428571]
"""