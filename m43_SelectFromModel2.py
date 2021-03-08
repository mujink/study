# 실습 
#  1. 상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2 값과 피처임포턴스 구할것

# 2 위 쓰레드 값으로 SelectFromModel을 구해서
#  최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2 구할 것

# 1번 값과 2번 값 비교



from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score



import warnings

warnings.filterwarnings('ignore')

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size=0.8, shuffle=True, random_state=66
)

print(x_train.shape) # 404,13
print(x_test.shape) # 102, 13

parameters = [
    {"n_estimators" : [100,200], "learning_rate" : [0.1,0.3,0.001,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate" : [0.1,0.001,0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators" : [90,110], "learning_rate" : [0.1,0.001,0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]},
]

# model = GridSearchCV(XGBRegressor(n_jobs=8,use_label_encoder=False), parameters)
model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters)

model.fit(x_train, y_train, eval_metric='logloss')
score = model.score(x_test,y_test)
print("R2 : ",score)
print(model.best_estimator_)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#              min_child_weight=1, min_samples_split=6, missing=nan,
#              monotone_constraints='()', n_estimators=100, n_job=8, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# R2 :  0.9221188601856797
# models = XGBRegressor(n_jobs=8)
models = model.best_estimator_
models.fit(x_train, y_train, eval_metric='logloss')
score = models.score(x_test,y_test)

thresholds = np.sort(models.feature_importances_)
print(thresholds)

# [0.00420776 0.00730204 0.01498673 0.01743406 0.02186794 0.02617752
#  0.0536502  0.05971902 0.07015063 0.07598229 0.14734462 0.19840336
#  0.3027738 ]

for thresh in thresholds:
    # selection = SelectFromModel(models, threshold=thresh, prefit=True)
    # select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)
    # selection_model = XGBRegressor(n_jobs=8)
    # selection_model.fit(select_x_train, y_train)
    # select_x_test = selection.transform(x_test)
    # y_predict = selection_model.predict(select_x_test)
    # score = r2_score(y_test, y_predict)
    # print("Thresh = %.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))
    selection = SelectFromModel(models, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters)
    selection_model.fit(select_x_train, y_train) 
    select_x_test = selection.transform(x_test)   
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh = %.3f, n=%d, R2: %.2f%%" %(thresh, x_train.shape[1], score*100))
"""

(404, 13)
Thresh = 0.001, n=13, R2: 92.21%
(404, 12)
Thresh = 0.004, n=12, R2: 92.16%
(404, 11)
Thresh = 0.012, n=11, R2: 92.03%
(404, 10)
Thresh = 0.012, n=10, R2: 92.19%
(404, 9)
Thresh = 0.014, n=9, R2: 93.08%  랜덤서치 이전의 최고 값
(404, 8)
Thresh = 0.015, n=8, R2: 92.37%
(404, 7)
Thresh = 0.018, n=7, R2: 91.48%
(404, 6)
Thresh = 0.030, n=6, R2: 92.71%
(404, 5)
Thresh = 0.042, n=5, R2: 91.74%
(404, 4)
Thresh = 0.052, n=4, R2: 92.11%
(404, 3)
Thresh = 0.069, n=3, R2: 92.52%
(404, 2)
Thresh = 0.301, n=2, R2: 69.41%
(404, 1)
Thresh = 0.428, n=1, R2: 44.98%


(404, 13)
Thresh = 0.003, n=13, R2: 92.21%
(404, 12)
Thresh = 0.004, n=12, R2: 91.96%
(404, 11)
Thresh = 0.010, n=11, R2: 92.03%
(404, 10)
Thresh = 0.012, n=10, R2: 92.00%
(404, 9)
Thresh = 0.013, n=9, R2: 93.08%
(404, 8)
Thresh = 0.015, n=8, R2: 93.52% 랜덤 서치 이후의 최고 값
(404, 7)
Thresh = 0.018, n=7, R2: 93.50%
(404, 6)
Thresh = 0.023, n=6, R2: 92.71%
(404, 5)
Thresh = 0.039, n=5, R2: 91.74%
(404, 4)
Thresh = 0.049, n=4, R2: 92.11%
(404, 3)
Thresh = 0.053, n=3, R2: 86.04%
(404, 2)
Thresh = 0.242, n=2, R2: 69.41%
(404, 1)
Thresh = 0.519, n=1, R2: 44.98%

컬럼 줄이고 랜덤서치 적용했을 때
Thresh = 0.006, n=13, R2: 93.50%
Thresh = 0.008, n=13, R2: 93.44%
Thresh = 0.021, n=13, R2: 93.54%
Thresh = 0.022, n=13, R2: 93.05%
Thresh = 0.022, n=13, R2: 93.72%
Thresh = 0.023, n=13, R2: 94.23% 최고 값 => 8 컬럼 파라미터는 ??
Thresh = 0.045, n=13, R2: 93.81%
Thresh = 0.058, n=13, R2: 92.86%
Thresh = 0.064, n=13, R2: 91.91%
Thresh = 0.090, n=13, R2: 93.44%
Thresh = 0.127, n=13, R2: 93.35%
Thresh = 0.215, n=13, R2: 93.05%
Thresh = 0.299, n=13, R2: 93.16%
"""