# m09 => Fold, cross_val_score

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
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
AdaBoostClassifier 의 정답률 :  [0.875      0.91666667 1.         0.95833333 0.875     ]
BaggingClassifier 의 정답률 :  [0.91666667 0.91666667 0.875      0.95833333 1.        ]
BernoulliNB 의 정답률 :  [0.375      0.33333333 0.29166667 0.41666667 0.125     ]
CalibratedClassifierCV 의 정답률 :  [0.91666667 0.95833333 0.875      0.75       0.95833333]
CategoricalNB 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ComplementNB 의 정답률 :  [0.875      0.66666667 0.625      0.625      0.66666667]
DecisionTreeClassifier 의 정답률 :  [0.95833333 0.91666667 0.95833333 1.         0.95833333]
DummyClassifier 의 정답률 :  [0.33333333 0.25       0.29166667 0.25       0.41666667]
ExtraTreeClassifier 의 정답률 :  [0.875      0.83333333 1.         0.875      0.875     ]
ExtraTreesClassifier 의 정답률 :  [1.         0.95833333 0.91666667 0.875      0.95833333]
GaussianNB 의 정답률 :  [0.95833333 0.875      0.95833333 0.95833333 0.95833333]
GaussianProcessClassifier 의 정답률 :  [0.91666667 0.95833333 0.875      1.         0.95833333]
GradientBoostingClassifier 의 정답률 :  [1.         0.91666667 0.95833333 1.         0.95833333]
HistGradientBoostingClassifier 의 정답률 :  [1.         0.95833333 0.83333333 0.91666667 0.95833333]
KNeighborsClassifier 의 정답률 :  [1.         0.91666667 0.95833333 0.95833333 0.95833333]
LabelPropagation 의 정답률 :  [0.875      0.95833333 1.         0.95833333 0.95833333]
LabelSpreading 의 정답률 :  [0.95833333 1.         0.95833333 0.91666667 0.91666667]
LinearDiscriminantAnalysis 의 정답률 :  [0.95833333 0.95833333 0.95833333 1.         1.        ]
LinearSVC 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.91666667 0.875     ]
LogisticRegression 의 정답률 :  [0.95833333 1.         0.95833333 1.         1.        ]
LogisticRegressionCV 의 정답률 :  [1.         1.         0.91666667 1.         0.95833333]
MLPClassifier 의 정답률 :  [0.91666667 0.95833333 1.         1.         1.        ]
MultiOutputClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultinomialNB 의 정답률 :  [0.70833333 0.66666667 0.79166667 0.54166667 0.91666667]
NearestCentroid 의 정답률 :  [0.875      0.91666667 0.95833333 0.95833333 0.91666667]
NuSVC 의 정답률 :  [0.95833333 1.         0.95833333 0.95833333 0.95833333]
OneVsOneClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OneVsRestClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OutputCodeClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
PassiveAggressiveClassifier 의 정답률 :  [0.83333333 0.95833333 0.66666667 0.83333333 0.75      ]
Perceptron 의 정답률 :  [0.625      1.         0.70833333 0.875      0.33333333]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.91666667 1.         1.         1.         1.        ]
RadiusNeighborsClassifier 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 0.95833333]
RandomForestClassifier 의 정답률 :  [0.95833333 0.95833333 0.91666667 0.95833333 0.95833333]
RidgeClassifier 의 정답률 :  [0.75       0.91666667 0.79166667 0.83333333 0.75      ]
RidgeClassifierCV 의 정답률 :  [0.75       0.91666667 0.83333333 0.83333333 0.91666667]
SGDClassifier 의 정답률 :  [0.91666667 0.79166667 0.70833333 0.79166667 0.70833333]
SVC 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 0.95833333]
StackingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
VotingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""