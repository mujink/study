from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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
ARDRegression 의 정답률 :  0.4282172805400202
AdaBoostRegressor 의 정답률 :  0.3874908939520475
BaggingRegressor 의 정답률 :  0.25576783571890915
BayesianRidge 의 정답률 :  0.43195678247082403
CCA 의 정답률 :  0.3457006535119874
DecisionTreeRegressor 의 정답률 :  -0.2989249616697387
DummyRegressor 의 정답률 :  -0.007154241099143865
ElasticNet 의 정답률 :  0.00159901087717218
ElasticNetCV 의 정답률 :  0.3984462985393632
ExtraTreeRegressor 의 정답률 :  -0.35707435736667614
ExtraTreesRegressor 의 정답률 :  0.32554722926189417
GammaRegressor 의 정답률 :  -0.00039758303797787775
GaussianProcessRegressor 의 정답률 :  -22.708730735470905
GeneralizedLinearRegressor 의 정답률 :  -0.0006935318688836567
GradientBoostingRegressor 의 정답률 :  0.2947750020727282
HistGradientBoostingRegressor 의 정답률 :  0.2912443692579407
HuberRegressor 의 정답률 :  0.4251214480910884
IsotonicRegression 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
KNeighborsRegressor 의 정답률 :  0.2574207258080603
KernelRidge 의 정답률 :  -4.011635425401908
Lars 의 정답률 :  0.4345357798835112
LarsCV 의 정답률 :  0.431915797073288
Lasso 의 정답률 :  0.31968708336543517
LassoCV 의 정답률 :  0.43216505374582537
LassoLars 의 정답률 :  0.33868045523477097
LassoLarsCV 의 정답률 :  0.4322934535818672
LassoLarsIC 의 정답률 :  0.4323680995024166
LinearRegression 의 정답률 :  0.43843604017332694
LinearSVR 의 정답률 :  -0.3153703417278835
MLPRegressor 의 정답률 :  -3.1135665875836844
MultiOutputRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNet 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNetCV 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskLasso 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskLassoCV 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
NuSVR 의 정답률 :  0.16738579518419927
OrthogonalMatchingPursuit 의 정답률 :  0.2512945795962108
OrthogonalMatchingPursuitCV 의 정답률 :  0.41739659135843676
PLSCanonical 의 정답률 :  -1.412090992523511
PLSRegression 의 정답률 :  0.4351260341539228
PassiveAggressiveRegressor 의 정답률 :  0.4063860725894973
PoissonRegressor 의 정답률 :  0.31157391160622183
RANSACRegressor 의 정답률 :  0.12926094999079862
RadiusNeighborsRegressor 의 정답률 :  -0.007154241099143865
RandomForestRegressor 의 정답률 :  0.2749980755482
RegressorChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Ridge 의 정답률 :  0.38597036901890236
RidgeCV 의 정답률 :  0.4327461382284624
SGDRegressor 의 정답률 :  0.37721302032630055
SVR 의 정답률 :  0.17710151701511323
StackingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
TheilSenRegressor 의 정답률 :  0.42215300548385337
TransformedTargetRegressor 의 정답률 :  0.43843604017332694
TweedieRegressor 의 정답률 :  -0.0006935318688836567
VotingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
_SigmoidCalibration 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""