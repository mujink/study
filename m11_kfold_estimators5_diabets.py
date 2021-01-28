# m09 => Fold, cross_val_score

from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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
ARDRegression 의 정답률 :  [0.48810158 0.48999351 0.61274573 0.41132578 0.35279514]
AdaBoostRegressor 의 정답률 :  [0.40677229 0.43192146 0.46205796 0.43763124 0.52557037]
BaggingRegressor 의 정답률 :  [0.42710117 0.41740943 0.236783   0.36018116 0.45215613]
BayesianRidge 의 정답률 :  [0.51491091 0.57291976 0.46952525 0.46569895 0.45809988]
CCA 의 정답률 :  [0.35629398 0.34307462 0.3561567  0.34151045 0.40686331]
DecisionTreeRegressor 의 정답률 :  [ 0.00704123 -0.08204229 -0.10543133 -0.31105362  0.02120625]
DummyRegressor 의 정답률 :  [-0.00366056 -0.01104296 -0.00262994 -0.00599055 -0.0044926 ]
ElasticNet 의 정답률 :  [-0.06083906  0.00665146 -0.05853191  0.00858917  0.00304163]
ElasticNetCV 의 정답률 :  [0.57381472 0.45081756 0.46497862 0.35319464 0.37483599]
ExtraTreeRegressor 의 정답률 :  [ 0.15130602 -0.19629602  0.17720134 -0.25362068  0.00177356]
ExtraTreesRegressor 의 정답률 :  [0.43869457 0.41266889 0.48415873 0.38115274 0.53317306]
GammaRegressor 의 정답률 :  [-0.03230559  0.00060196 -0.0271027  -0.0059648   0.00615249]
GaussianProcessRegressor 의 정답률 :  [-16.19029544 -13.03694327  -8.5290114  -19.90843405 -10.15998632]
GeneralizedLinearRegressor 의 정답률 :  [ 0.0051747  -0.00832899  0.00679053  0.00430002  0.00642891]
GradientBoostingRegressor 의 정답률 :  [0.39910772 0.4462177  0.40229771 0.66141522 0.34063587]
HistGradientBoostingRegressor 의 정답률 :  [0.6115256  0.47563223 0.35827286 0.32005962 0.46711251]
HuberRegressor 의 정답률 :  [0.53477085 0.38074611 0.52491316 0.63508297 0.37308363]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.46832765 0.04317218 0.52716383 0.44457878 0.31811   ]
KernelRidge 의 정답률 :  [-4.04188167 -3.2630171  -3.41257969 -3.52946669 -3.06351309]
Lars 의 정답률 :  [0.098874   0.16821673 0.30094166 0.50610595 0.54764777]
LarsCV 의 정답률 :  [0.50472044 0.59376654 0.35334315 0.49584133 0.45422329]
Lasso 의 정답률 :  [0.32061816 0.35949229 0.40025329 0.35732649 0.35954862]
LassoCV 의 정답률 :  [0.52989479 0.48527868 0.53960545 0.45031929 0.52797951]
LassoLars 의 정답률 :  [0.42173019 0.42584656 0.42439975 0.37385067 0.27701832]
LassoLarsCV 의 정답률 :  [0.55323655 0.54756437 0.30907848 0.56683504 0.52081098]
LassoLarsIC 의 정답률 :  [0.49794937 0.59433892 0.56701562 0.28411414 0.47189042]
LinearRegression 의 정답률 :  [0.57161094 0.46511594 0.52147859 0.5699174  0.37506528]
LinearSVR 의 정답률 :  [-0.42613014 -0.45560547 -0.34816873 -0.58123826 -0.48898941]
MLPRegressor 의 정답률 :  [-3.82627641 -3.97960179 -2.58066608 -2.43834094 -2.69217606]
MultiOutputRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.11251325 0.02003996 0.11564836 0.05199369 0.09117097]
OrthogonalMatchingPursuit 의 정답률 :  [0.3314335  0.25332076 0.22677365 0.24990304 0.40006541]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.51870882 0.52034106 0.43588092 0.39730976 0.54893768]
PLSCanonical 의 정답률 :  [-1.7329323  -1.00039158 -0.93334793 -0.96556631 -1.16065713]
PLSRegression 의 정답률 :  [0.42826868 0.5121055  0.49660164 0.54139161 0.52803872]
PassiveAggressiveRegressor 의 정답률 :  [0.53052786 0.38879859 0.55010392 0.32541902 0.44555581]
PoissonRegressor 의 정답률 :  [0.39586418 0.26228657 0.32178643 0.34245767 0.33103292]
RANSACRegressor 의 정답률 :  [-0.07329093  0.19660231  0.51624332  0.09149269 -0.4897754 ]
RadiusNeighborsRegressor 의 정답률 :  [-9.20525497e-03 -7.76932786e-06 -4.07545767e-03 -5.98744618e-03
 -2.69086428e-03]
RandomForestRegressor 의 정답률 :  [0.32550791 0.40829568 0.50567687 0.44298189 0.54226967]
RegressorChain 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Ridge 의 정답률 :  [0.5176657  0.36940652 0.33184928 0.42768517 0.45670899]
RidgeCV 의 정답률 :  [0.59912643 0.40666598 0.50453085 0.47456599 0.50841084]
SGDRegressor 의 정답률 :  [0.44893475 0.36953819 0.37019108 0.40526921 0.45875352]
SVR 의 정답률 :  [0.12265253 0.14422378 0.10586761 0.13722273 0.12900997]
StackingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
TheilSenRegressor 의 정답률 :  [0.54081796 0.54280968 0.50456776 0.47736554 0.4677277 ]
TransformedTargetRegressor 의 정답률 :  [0.57308246 0.47535177 0.41668188 0.59032944 0.43757169]
TweedieRegressor 의 정답률 :  [-0.00287619  0.00709629 -0.08326819 -0.00346178 -0.00881024]
VotingRegressor 은 없는 놈!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
"""