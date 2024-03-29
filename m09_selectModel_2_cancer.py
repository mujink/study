from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 :  0.9298245614035088
BaggingClassifier 의 정답률 :  0.9385964912280702
BernoulliNB 의 정답률 :  0.631578947368421
CalibratedClassifierCV 의 정답률 :  0.8859649122807017
CategoricalNB 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CheckingClassifier 의 정답률 :  0.3684210526315789
ClassifierChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ComplementNB 의 정답률 :  0.868421052631579
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.43859649122807015
ExtraTreeClassifier 의 정답률 :  0.9210526315789473
ExtraTreesClassifier 의 정답률 :  0.956140350877193
GaussianNB 의 정답률 :  0.9473684210526315
GaussianProcessClassifier 의 정답률 :  0.956140350877193
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
HistGradientBoostingClassifier 의 정답률 :  0.9473684210526315
KNeighborsClassifier 의 정답률 :  0.9385964912280702
LabelPropagation 의 정답률 :  0.42105263157894735
LabelSpreading 의 정답률 :  0.42105263157894735
LinearDiscriminantAnalysis 의 정답률 :  0.9473684210526315
LinearSVC 의 정답률 :  0.8947368421052632
LogisticRegression 의 정답률 :  0.9473684210526315
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9385964912280702
MultiOutputClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultinomialNB 의 정답률 :  0.868421052631579
NearestCentroid 의 정답률 :  0.8596491228070176
NuSVC 의 정답률 :  0.8421052631578947
OneVsOneClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OneVsRestClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OutputCodeClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
PassiveAggressiveClassifier 의 정답률 :  0.8596491228070176
Perceptron 의 정답률 :  0.8508771929824561
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RadiusNeighborsClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
RandomForestClassifier 의 정답률 :  0.9473684210526315
RidgeClassifier 의 정답률 :  0.9385964912280702
RidgeClassifierCV 의 정답률 :  0.9385964912280702
SGDClassifier 의 정답률 :  0.9298245614035088
SVC 의 정답률 :  0.9035087719298246
StackingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
VotingClassifier 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""