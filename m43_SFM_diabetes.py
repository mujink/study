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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score



import warnings

warnings.filterwarnings('ignore')

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size=0.8, shuffle=True, random_state=66
)

print(x_train.shape) # 353, 10
print(x_test.shape) # 89, 10

parameters = [
    {"n_estimators" : [200,300], "learning_rate" : [0.1,0.3,0.001,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate" : [0.1,0.001,0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators" : [90,110], "learning_rate" : [0.1,0.001,0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]},
]

# model = GridSearchCV(XGBRegressor(n_jobs=8,use_label_encoder=False), parameters)
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters)
resulet = cross_val_score(model, x_train, y_train, cv=kfold)

# model = GridSearchCV(SVC(), parameters, cv=kfold)
# score = cross_val_score(model, x_train, y_train, cv=kfold)
# model.fit(x_train, y_train,eval_metric='mlogloss')
score = model.score(x_test,y_test)
print("R2 : ",score)
# print(model.best_estimator_)

# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# R2 :  0.33518059128246014
models = model.best_estimator_
models.fit(x_train, y_train, eval_metric='mlogloss')
score = models.score(x_test,y_test)

thresholds = np.sort(models.feature_importances_)
print(thresholds)

# [0.04241714 0.0489521  0.04906641 0.05020966 0.0505336  0.0628658
#  0.07641661 0.08562486 0.17060308 0.36331072]

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
    print(model.best_estimator_)
    print("Thresh = %.3f, n=%d, R2: %.2f%%" %(thresh, x_train.shape[1], score*100))
"""
(353, 10)
Thresh = 0.042, n=10, R2: 27.43%
(353, 9)
Thresh = 0.049, n=10, R2: 40.85%
(353, 8)
Thresh = 0.049, n=10, R2: 32.11%
(353, 7)
Thresh = 0.050, n=10, R2: 38.06%
(353, 6)
Thresh = 0.051, n=10, R2: 35.04%
(353, 5)
Thresh = 0.063, n=10, R2: 34.97%
(353, 4)
Thresh = 0.076, n=10, R2: 41.10%
(353, 3)
Thresh = 0.086, n=10, R2: 37.71%
(353, 2)
Thresh = 0.171, n=10, R2: 36.85%
(353, 1)
Thresh = 0.363, n=10, R2: 21.08%
"""