# m09 => Fold, cross_val_score

from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')

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
ARDRegression 의 정답률 :  [0.65944414 0.76466833 0.64851608 0.62111638 0.72163851]
AdaBoostRegressor 의 정답률 :  [0.78064005 0.7818456  0.79412491 0.89854943 0.85772276]
BaggingRegressor 의 정답률 :  [0.88481555 0.81630159 0.88656936 0.71200454 0.88695661]
BayesianRidge 의 정답률 :  [0.69320294 0.74054816 0.72482959 0.5934829  0.67163548]
CCA 의 정답률 :  [0.70333219 0.61022009 0.55676445 0.74532334 0.61771745]
DecisionTreeRegressor 의 정답률 :  [0.84634739 0.72544958 0.51614476 0.71560465 0.72613295]
DummyRegressor 의 정답률 :  [-1.45538926e-02 -1.17682342e-05 -4.61274449e-03 -4.46987385e-03
 -8.54381631e-03]
ElasticNet 의 정답률 :  [0.62303796 0.6216945  0.60384415 0.71413885 0.73356588]
ElasticNetCV 의 정답률 :  [0.65299241 0.69810532 0.61281286 0.60013149 0.68315769]
ExtraTreeRegressor 의 정답률 :  [0.58639288 0.81084054 0.56858283 0.54249203 0.18185226]
ExtraTreesRegressor 의 정답률 :  [0.8846784  0.83289653 0.90627541 0.74584498 0.85735352]
GammaRegressor 의 정답률 :  [-0.00836627 -0.01785519 -0.01768276 -0.00240739 -0.00075614]
GaussianProcessRegressor 의 정답률 :  [-5.46984549 -8.03442923 -5.54407103 -7.60426746 -6.25091128]
GeneralizedLinearRegressor 의 정답률 :  [0.61711775 0.72983599 0.57553048 0.60963233 0.65152874]
GradientBoostingRegressor 의 정답률 :  [0.91114259 0.90527498 0.83769057 0.83848339 0.89424786]
HistGradientBoostingRegressor 의 정답률 :  [0.89620515 0.8364972  0.74427226 0.86271851 0.81900779]
HuberRegressor 의 정답률 :  [0.66935424 0.66467543 0.66262689 0.48365409 0.69606189]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.54016809 0.2917308  0.47699689 0.28324167 0.51816186]
KernelRidge 의 정답률 :  [0.66118477 0.61767995 0.74967757 0.46450891 0.76861601]
Lars 의 정답률 :  [0.64661201 0.7662253  0.71749602 0.69251286 0.61651989]
LarsCV 의 정답률 :  [0.71528766 0.69673443 0.5554649  0.69974953 0.76737495]
Lasso 의 정답률 :  [0.7354532  0.49842563 0.65847537 0.43808748 0.68109208]
LassoCV 의 정답률 :  [0.73175247 0.7192839  0.54828362 0.68908255 0.59186247]
LassoLars 의 정답률 :  [-0.10062323 -0.01018333 -0.00065988 -0.02384719 -0.00102191]
LassoLarsCV 의 정답률 :  [0.68240513 0.76260559 0.71755028 0.54268366 0.75006421]
LassoLarsIC 의 정답률 :  [0.75544268 0.73169874 0.72818995 0.6867903  0.61952248]
LinearRegression 의 정답률 :  [0.65376551 0.70353647 0.56936482 0.67029376 0.77951453]
LinearSVR 의 정답률 :  [ 0.50197337  0.59868356 -0.02421178 -0.71257026 -0.76551574]
MLPRegressor 의 정답률 :  [0.63806365 0.54177978 0.42464839 0.51548704 0.65066762]
MultiOutputRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.17424139 0.32876666 0.12048167 0.21378751 0.25982291]
OrthogonalMatchingPursuit 의 정답률 :  [0.59209653 0.48070725 0.47718679 0.61742101 0.5325491 ]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.68561317 0.62908066 0.45074669 0.73262453 0.7184416 ]
PLSCanonical 의 정답률 :  [-1.73707605 -2.94285352 -2.646341   -1.64957207 -2.18571835]
PLSRegression 의 정답률 :  [0.71640405 0.70502308 0.43767821 0.70474889 0.82210633]
PassiveAggressiveRegressor 의 정답률 :  [ 0.38578366 -0.59518591  0.09701235  0.16314869  0.15155382]
PoissonRegressor 의 정답률 :  [0.68418882 0.77855851 0.71037184 0.74234761 0.73746289]
RANSACRegressor 의 정답률 :  [ 0.38384903  0.70415282  0.67064401 -0.04275375  0.55745816]
RadiusNeighborsRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
RandomForestRegressor 의 정답률 :  [0.87375486 0.86432058 0.88595821 0.76265687 0.85646967]
RegressorChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Ridge 의 정답률 :  [0.64487006 0.64877383 0.7908231  0.75875285 0.70546922]
RidgeCV 의 정답률 :  [0.74632865 0.71443241 0.5589111  0.73567386 0.68575066]
SGDRegressor 의 정답률 :  [-1.00543701e+27 -4.61245106e+26 -7.95749276e+24 -3.91286807e+25
 -2.53430000e+25]
SVR 의 정답률 :  [0.2576072  0.12868065 0.12442388 0.25757101 0.33397873]
StackingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
TheilSenRegressor 의 정답률 :  [0.51705634 0.64373125 0.77325224 0.70318495 0.69693804]
TransformedTargetRegressor 의 정답률 :  [0.6261554  0.76937851 0.66523652 0.6820863  0.77033011]
TweedieRegressor 의 정답률 :  [0.56755253 0.65421261 0.57555033 0.72557492 0.60952135]
VotingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
"""