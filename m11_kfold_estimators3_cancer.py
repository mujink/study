# m09 => Fold, cross_val_score

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 :  [0.97802198 0.93406593 0.97802198 0.95604396 0.94505495]
BaggingClassifier 의 정답률 :  [0.97802198 0.96703297 0.93406593 0.95604396 0.94505495]
BernoulliNB 의 정답률 :  [0.69230769 0.59340659 0.57142857 0.64835165 0.62637363]
CalibratedClassifierCV 의 정답률 :  [0.85714286 0.94505495 0.96703297 0.93406593 0.93406593]
CategoricalNB 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ComplementNB 의 정답률 :  [0.91208791 0.91208791 0.86813187 0.91208791 0.91208791]
DecisionTreeClassifier 의 정답률 :  [0.96703297 0.9010989  0.92307692 0.91208791 0.94505495]
DummyClassifier 의 정답률 :  [0.47252747 0.52747253 0.54945055 0.50549451 0.53846154]
ExtraTreeClassifier 의 정답률 :  [0.92307692 0.96703297 0.96703297 0.93406593 0.91208791]
ExtraTreesClassifier 의 정답률 :  [0.98901099 0.93406593 0.94505495 0.95604396 0.96703297]
GaussianNB 의 정답률 :  [0.92307692 0.93406593 0.92307692 0.93406593 0.95604396]
GaussianProcessClassifier 의 정답률 :  [0.9010989  0.89010989 0.89010989 0.91208791 0.89010989]
GradientBoostingClassifier 의 정답률 :  [0.94505495 0.98901099 0.94505495 0.91208791 0.96703297]
HistGradientBoostingClassifier 의 정답률 :  [0.96703297 0.95604396 0.98901099 0.97802198 0.94505495]
KNeighborsClassifier 의 정답률 :  [0.92307692 0.91208791 0.89010989 0.92307692 0.92307692]
LabelPropagation 의 정답률 :  [0.40659341 0.35164835 0.35164835 0.34065934 0.43956044]
LabelSpreading 의 정답률 :  [0.3956044  0.38461538 0.43956044 0.40659341 0.28571429]
LinearDiscriminantAnalysis 의 정답률 :  [0.94505495 0.96703297 0.96703297 0.94505495 0.95604396]
LinearSVC 의 정답률 :  [0.92307692 0.92307692 0.93406593 0.86813187 0.94505495]
LogisticRegression 의 정답률 :  [0.93406593 0.93406593 0.95604396 0.96703297 0.94505495]
"""