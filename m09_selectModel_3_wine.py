from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        # continue
        print(name, "은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
import sklearn
print(sklearn.__version__) # 0.23.2

"""
AdaBoostClassifier 의 정답률 :  0.8333333333333334
BaggingClassifier 의 정답률 :  0.9444444444444444
BernoulliNB 의 정답률 :  0.3611111111111111
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CheckingClassifier 의 정답률 :  0.3888888888888889
ClassifierChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ComplementNB 의 정답률 :  0.7222222222222222
DecisionTreeClassifier 의 정답률 :  0.9166666666666666
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.8055555555555556
ExtraTreesClassifier 의 정답률 :  0.9722222222222222
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.4166666666666667
GradientBoostingClassifier 의 정답률 :  0.9444444444444444
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.6388888888888888
LabelPropagation 의 정답률 :  0.4722222222222222
LabelSpreading 의 정답률 :  0.4722222222222222
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.6944444444444444
LogisticRegression 의 정답률 :  0.9444444444444444
LogisticRegressionCV 의 정답률 :  0.9722222222222222
MLPClassifier 의 정답률 :  0.8611111111111112
MultiOutputClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultinomialNB 의 정답률 :  0.8333333333333334
NearestCentroid 의 정답률 :  0.6666666666666666
NuSVC 의 정답률 :  0.8611111111111112
OneVsOneClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OneVsRestClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OutputCodeClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
PassiveAggressiveClassifier 의 정답률 :  0.6388888888888888
Perceptron 의 정답률 :  0.5277777777777778
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
RandomForestClassifier 의 정답률 :  0.9722222222222222
RidgeClassifier 의 정답률 :  0.9722222222222222
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.5555555555555556
SVC 의 정답률 :  0.6388888888888888
StackingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
VotingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""