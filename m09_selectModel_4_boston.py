from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test,y_pred))
    except:
        # continue
        print(name, "은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
import sklearn
print(sklearn.__version__) # 0.23.2

"""
ARDRegression 의 정답률 :  0.7385855359764906
AdaBoostRegressor 의 정답률 :  0.8546117601099857
BaggingRegressor 의 정답률 :  0.8717387102915837
BayesianRidge 의 정답률 :  0.7489159523194986
CCA 의 정답률 :  0.7749569424747094
DecisionTreeRegressor 의 정답률 :  0.8239483448436733
DummyRegressor 의 정답률 :  -2.7606132582347342e-05
ElasticNet 의 정답률 :  0.6662534357446656
ElasticNetCV 의 정답률 :  0.6465211400827211
ExtraTreeRegressor 의 정답률 :  0.7339548361867596
ExtraTreesRegressor 의 정답률 :  0.9132042682234167
GammaRegressor 의 정답률 :  -2.7606132582569387e-05
GaussianProcessRegressor 의 정답률 :  -4.904527258611498
GeneralizedLinearRegressor 의 정답률 :  0.6684375377293572
GradientBoostingRegressor 의 정답률 :  0.921606590657268
HistGradientBoostingRegressor 의 정답률 :  0.8970343402882811
HuberRegressor 의 정답률 :  0.6467229926598166
IsotonicRegression 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
KNeighborsRegressor 의 정답률 :  0.5401612153026705
KernelRidge 의 정답률 :  0.779674680277358
Lars 의 정답률 :  0.7621351463298283
LarsCV 의 정답률 :  0.71325183158015
Lasso 의 정답률 :  0.6399927356461494
LassoCV 의 정답률 :  0.6837946514509451
LassoLars 의 정답률 :  -2.7606132582347342e-05
LassoLarsCV 의 정답률 :  0.7618367513678929
LassoLarsIC 의 정답률 :  0.7622134786227756
LinearRegression 의 정답률 :  0.7634174432138463
LinearSVR 의 정답률 :  0.18393873372454161
MLPRegressor 의 정답률 :  0.4138912855645873
MultiOutputRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNet 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNetCV 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskLasso 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskLassoCV 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
NuSVR 의 정답률 :  0.22704921927288402
OrthogonalMatchingPursuit 의 정답률 :  0.5244757432765152
OrthogonalMatchingPursuitCV 의 정답률 :  0.6959056368091425
PLSCanonical 의 정답률 :  -1.3929907067150258
PLSRegression 의 정답률 :  0.7574698510203197
PassiveAggressiveRegressor 의 정답률 :  0.23834136767028713
PoissonRegressor 의 정답률 :  0.8313511205762624
RANSACRegressor 의 정답률 :  0.5738183233780103
RadiusNeighborsRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
RandomForestRegressor 의 정답률 :  0.9074126922602308
RegressorChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Ridge 의 정답률 :  0.7655800611077145
RidgeCV 의 정답률 :  0.7641649997446235
SGDRegressor 의 정답률 :  -1.0418138405109048e+26
SVR 의 정답률 :  0.18157166564230964
StackingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
TheilSenRegressor 의 정답률 :  0.7491795511698468
TransformedTargetRegressor 의 정답률 :  0.7634174432138463
TweedieRegressor 의 정답률 :  0.6684375377293572
VotingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
_SigmoidCalibration 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""