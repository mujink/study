# 모델 랜덤포레스트 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 디아벳

import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()

x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

# 딕셔너리 키벨류 리스트 형식
parameters = [
    {"Rand__n_estimators" : [1, 10, 100], "Rand__min_samples_split" : [6,8,10]},
    {"Rand__n_estimators" : [2, 200], "Rand__min_samples_split" : [2,4,6], "Rand__max_depth" : [10,100]    },
    {"Rand__n_estimators" : [3, 10, 300], "Rand__min_samples_split" : [12,14,16], "Rand__criterion" : ["gini","entropy"]}
]

pips = [Pipeline, make_pipeline]
proprecess = [MinMaxScaler, StandardScaler]
model = [RandomizedSearchCV, GridSearchCV]

# 모델, 파라미터, 교차검증

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for pro in proprecess:
        for pip in pips:
            print("pip :",pip)
            # make_pipeline 은 <function make_pipeline at 0x00000286DE3C5310> 로 확인됨
            # make_pipeline 은 Pipeline 로 인식함
            # make_pipeline 은  키워드 없는 형태의 인풋을 Pipeline 형식으로 키워드를 입혀 Pipeline에 인풋 값을 리턴하는 함수임
            # 만약 키워드가 있으면 그대로 Pipeline에 반환하기 때문에 잘 돌아감. 
            if pip == Pipeline:
                pipe = pip([("scaler", pro()),("Rand", RandomForestRegressor())])
            for Search in model:
                print("서치 : ",Search)
                # 가장 잘나온 파라미터를 서치해서 크로스 발리데이션 스코어로 전달해줌
                models = Search(pipe,parameters, cv=kfold )
                score = cross_val_score(models, x_train, y_train, cv=kfold)
                print("교차검증 점수", score)
                


# 교차 검증이 총 40번 나와야함 아래 포문 8줄씩 * 트레인 5셋 을 돌렸음
# pips = [Pipeline, make_pipeline]
# proprecess = [MinMaxScaler, StandardScaler]
# model = [RandomizedSearchCV, GridSearchCV]
"""
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.44420708 0.2954477  0.49560638 0.51370507 0.38956753]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.35690067 0.48093287 0.50847007 0.48595785 0.21470615]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.45635761 0.30192216 0.36938819 0.49756138 0.45560358]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.29945314 0.38040816 0.41737985 0.46029245 0.54640974]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.36145683 0.53940694 0.49841249 0.42135903 0.29578018]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.45474218 0.41890813 0.46335879 0.2894753  0.30902481]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.55151745 0.35948962 0.32001508 0.31007525 0.45064292]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.46662537 0.37965    0.34004806 0.44958776 0.42958446]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.44713099 0.4145787  0.36433485 0.28582927 0.37360183]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.42136449 0.38822089 0.4640027  0.48616672 0.38256604]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.26637478 0.54433415 0.37043374 0.32168608 0.42111619]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.3318184  0.46866888 0.40562807 0.3932662  0.45962407]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.22726355 0.48606629 0.36312535 0.50455088 0.46111959]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.37805092 0.5216449  0.38493368 0.33746105 0.48354669]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.37447958 0.40057859 0.38343111 0.46176696 0.51457653]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.31262501 0.3449527  0.32639277 0.55169665 0.49013033]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.31608819 0.44940146 0.49440692 0.44081886 0.48430412]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.31854616 0.32569214 0.3600241  0.56092579 0.46627624]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.32633219 0.53792115 0.43092839 0.43246974 0.35527982]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.45841474 0.35953431 0.62357117 0.28304623 0.4638477 ]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.35726424 0.37799358 0.470823   0.27962394 0.47005707]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.15446194 0.42889316 0.34320298 0.55654715 0.37061158]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.40817922 0.34804727 0.46057154 0.35936025 0.38657004]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.38361091 0.39941683 0.54292417 0.51971677 0.34914707]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.44704079 0.18796435 0.34098254 0.4708168  0.47664789]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.5269942  0.52373225 0.35719235 0.32704845 0.3891933 ]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.40117601 0.44787021 0.45782445 0.39900978 0.40207198]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.34780868 0.41140913 0.44655594 0.48877041 0.57363007]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.42216533 0.35788206 0.41382631 0.4471961  0.48670448]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.33341517 0.20436778 0.55507598 0.59543464 0.50135058]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.54641014 0.43937494 0.49419782 0.3908315  0.37733412]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.41028103 0.57947757 0.41829064 0.30035547 0.41343455]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.34769659 0.61106789 0.42669694 0.50205942 0.32293402]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.51540485 0.40422884 0.36088307 0.48482735 0.45201977]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.49708667 0.45299515 0.37342653 0.43704345 0.38927646]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.3699202  0.43052778 0.53467916 0.45596308 0.46284664]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.5142623  0.38092803 0.58939341 0.4599742  0.22086183]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.47644167 0.46819671 0.43944681 0.38344239 0.46882971]
pip : <function make_pipeline at 0x00000279A9F3B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.48638778 0.30750001 0.51247366 0.25058426 0.46609773]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.43649259 0.3444968  0.44763764 0.40077185 0.56444794]
"""