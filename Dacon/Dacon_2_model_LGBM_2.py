# Dense 모델이 더 잘나왔음
# Dacon_1_model_dense

"""
Dacon_2_model_LGBM 베이스에서 트레인의 컬럼 값을 수정함.
별 차이 없거나 떨어짐.
"""


import pandas as pd
import numpy as np

#  이것들은 뭐하는 라이브러리 인지 모르겟음.

# 운영체제에서 할 수 있는 일들을 제어할 수 있음.
# 파일이나 폴더 삭제 수정, 인터럭트를 걸 수 있다고함. (인터럭트는 또 뭐야)
# 없어도 돌아가지 않을까 싶은데 아직 잘모르겠음.
import os

# 이걸 쓰면 폴더에 있는 경로를 그대로 가져다 써도 되는 듯(?)
# 파일을 리스트로 바꾼다던지 csv 파일 외에도 다른 확장자를 가진 파일을 엑세스 하는 듯.
import glob

# 랜덤 값을 돌려주는 모듈임 왜 있는지 모르겠음.==================================표시
import random

# 파이썬 경고 메시지나 뭐였더라 여튼 그런거 제어하려고 함.
import warnings

# 일치하는 경고를 표시하지 않는 경고필터 적용됨.
warnings.filterwarnings("ignore")

# 트레인 파일 로드함
train = pd.read_csv('../data/csv/Dacon/train/train.csv')

# 트레인 파일의 끝부분 5개를 봄
print(train.tail())
"""
        Day  Hour  Minute  DHI  DNI   WS     RH  T  TARGET
52555  1094    21      30    0    0  2.4  70.70 -4     0.0
52556  1094    22       0    0    0  2.4  66.79 -4     0.0
52557  1094    22      30    0    0  2.2  66.78 -4     0.0
52558  1094    23       0    0    0  2.1  67.72 -4     0.0
52559  1094    23      30    0    0  2.1  67.70 -4     0.0
"""
# 제출할 파일 양식을 로드함
submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

# 제출할 파일 양식의 끝 5열을 살벼봄
print(submission.tail())

"""
                      id  q_0.1  q_0.2  q_0.3  q_0.4  q_0.5  q_0.6  q_0.7  q_0.8  q_0.9
7771  80.csv_Day8_21h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
7772  80.csv_Day8_22h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
7773  80.csv_Day8_22h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
7774  80.csv_Day8_23h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
7775  80.csv_Day8_23h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
"""
#  데이터와 잇 트레인을 인자로 갖는 함수를 정의함 
#  함수가 호출 될 때 인자 잇 트레인의 값이 주어지지 않으면 디폴트로 트루를 가지게 됨.
def preprocess_data(data, is_train=True):
    
    # 템프는 데이터 인자의 값을 복사하여 초기화 됨
    temp = data.copy()
    
    # 템프는 아래 컬럼의 제외한 나머지 컬럼을 버림
    temp = temp[['Day', 'Minute', 'Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    #  만약 잇 트레인의 값이 트루일 때
    if is_train==True:          
        
        # 템프는 타겟1 이라는 컬럼을 갖게됨
        # 타겟1 이라는 컬럼은 다음 날의 타겟의 값으로 초기화됨
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')

        # 타겟2 라는 컬럼도 같게됨
        # 타겟2의 컬럼엔 다 다음 날의 타겟의 값이 입혀짐
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        # 그리고 마지막 2일치를 제외한 값을 호출된 위치에 반환하고 함수를 종료함.
        return temp.iloc[:-96]

    #  만약 잇 트레인이 트루가 아닌 펠스 일 때
    elif is_train==False:
        
        # 위에서 컬럼 버린 걸 또 확인하고 (왜지??)
        temp = temp[['Day', 'Minute', 'Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        # 맨 밑줄 부터 하루치 행을 함수가 호출된 위치에 반환하고 함수가 종료됨.
        return temp.iloc[-48:, :]

# 전각 일사량, 이슬점 온도 구하는 공식임=====  이거 추가함
def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

#  로드한 트레인 데이터를 함수에 호출함.
#  잇 트레인의 값을 지정하지 않았으니 잇 트레인은 디폴트 값인 투르를 가짐.
df_train = preprocess_data(train)
df_train = Add_features(df_train)
# 전처리된 트레인 데이터를 인덱스 넘버 로케이션으로 첫 하루치를 확인함.
print(df_train.iloc[:48])

# 전처리전의 로드된 트레인 데이터의 위에서 2일치 분량을 확인함.
print(train.iloc[48:96])

# 전처리전의 로드된 트레인 데이터의 3일치 분량을 확인함.
print(train.iloc[48+48:96+48])

# 전처리된 트레인 데이터의 마지막 5일을 확인함
print(df_train.tail())

# 테스트 파일 합치는 듯
df_test = []

# 파일수가 81개 0부터 80까지임 그걸 로드함
for i in range(81):
    # 파일 이름이 0~80이라 포문써서 돌림
    file_path = '../data/csv/Dacon/test/' + str(i) + '.csv'
    # 파일 로드
    temp = pd.read_csv(file_path)
    # 전처리함
    temp = preprocess_data(temp, is_train=False)
    temp = Add_features(temp)

    # 리스트에 추가함
    df_test.append(temp)

# 테스트 데이터를 엑스 테스트랑 합침
X_test = pd.concat(df_test)

print(X_test.shape)

print(X_test.head(48))

print(df_train.head())

print(df_train.iloc[-48:])

"""
여기서 셋의 컬럼을 자르고 구성함
"""
# del df_train['Hour']
# del df_train['T']
# del df_train['Td']
# del df_train['T-Td']
# del df_train['WS']
# del df_train['RH']
# del df_train['DNI']
# del df_train['DHI']
# del df_train['TARGET']
# del df_train['GHI']

# del X_test['Hour']
# del X_test['T']
# del X_test['Td']
# del X_test['T-Td']
# del X_test['WS']
# del X_test['RH']
# del X_test['DNI']
# del X_test['DHI']
# del X_test['TARGET']
# del X_test['GHI']

# 스플릿함
# 위에서 전처리한 엑스 트레인만 가지고 길이를 바꿔서 패밀리 1, 2를 만듬.
# 마지막 타겟 1,2를 제외한 모든 컬럼을 핏할 때 사용하고
# 패밀리 1 의 출력을 1일 후의 타겟으로 함.
# 패밀리 2 의 출력을 2일 후의 타겟으로 함.
from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

print(X_train_1.head())

print(X_test.head())

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler() 
scaler = StandardScaler() 
scaler.fit(X_train_1)
X_train_1 = scaler.transform(X_train_1)
X_valid_1 = scaler.transform(X_valid_1)
X_train_2 = scaler.transform(X_train_2)
X_valid_2 = scaler.transform(X_valid_2)
X_test = scaler.transform(X_test)

# =============================================================
import matplotlib.pyplot as plt
#       Hour  GHI       T-Td         Td     TARGET  DHI  DNI   WS     RH  T    Target1    Target2
plt.figure(figsize=(10,6))

# hist : 
#       data : 데이터 프레임.
#       x : 데이터 프레임의 column.
plt.subplot(2,2,1)
plt.hist(x = 'Hour', data=X_train_1)
plt.title('Hour')

plt.subplot(2,2,2)
plt.hist(x = 'GHI', data=X_train_1)
plt.title('GHI')

plt.subplot(2,2,3)
plt.hist(x = 'DHI', data=X_train_1)
plt.title('DHI')

plt.subplot(2,2,4)
plt.hist(x = 'DNI', data=X_train_1)
plt.title('DNI')
plt.show()
# =============================================================


# 분위수
# 크기 순으로 나열한 시리얼 자료를 전체를 100%로 보고 데이터의 분포를 10% 단위로 나누어서 그 퀀타일의 분포에 위치한 값이 무엇일까라는 거임
# 도수 분포 표에서 엑스 축에 해당하는 눈금에 포함된 데이터 양을 볼 수 있음.
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 모델 인듯
# 사이킷런의 에이피아이라고 함 라이트쥐비엠의 회귀모델임.
from lightgbm import LGBMRegressor #다운 받아야할 듯 이놈만 일단 해결해본다.


# Get the model and the predictions in (a) - (b)
# 예측 모델을 때마다 불러와서 쓸 모양임.
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    # 인자 q 는 분위수가 될 듯

    # 모델, 핏, 프래딕, 프래딕 리턴으로 구성됨.
    # 근데 어디를 봐도 컴파일이 없음.

    # (a) Modeling 
    # 모델 구성을 함수로 잡음. 오브젝티브, 알파, 에스트메이터스 기타 등등.
    # LGBMRegressor 는 회귀모델임
    # 아직 모르겠음 나중에 볼래===============================================================================표시
    # objective='quantile' 퀀타일 로스의 를 오브젝티브 메소드에서 지원함
    # alpha=q 분위수를 입력함
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
    # 함수안에는 모델 구성이랑  이벨류 메트릭, 셋 얼리스탑 랜덤, 벌보스 500
    # 아직 모르겠음 나중에 볼래===============================================================================표시
    # 평가 값을 퀀타일 로스로 확인함 테스트를 이벨류 셋으로 넣고 얼리스탑
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    # 엑스 프래딕을 랜덤 2해서 시리얼로 저장하고 리턴하는 듯
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model




# Target 예측
# 이제 트레인 데이터로 모델을 핏하고 결과를 보려하는 듯
def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    # 무언가 데이터 프레임이랑 리스트를 생성함.
    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    # 모델에 들어가는 큐가 for 문에 있었음 게다가 분위수임
    for q in quantiles:
        
        # 트레인 데이터로 모델을 총 분위수의 수 만큼 총 9번 돌리는데 진행사항을 보려는 듯 
        print(q)

        # 위에서 만든 모델 디파인을 트레인 이벨류에서 불러옴
        # 아까 구성이 모델링이랑 핏이랑 이벨류해서 엑스프레딕이랑 모델을 리턴하는 거였음.
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)

        # 모델을 리스트에 담음
        LGBM_models.append(model)
        # 알지비엠 엑튜얼 프래딕이랑 프래딕이랑 붙임 팬파인애플애플팬 그걸 알지비엠 엑튜얼로 초기화함(왜 이렇게하지?)
        # 알지비엠은 모델의 프래딕이 됨
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    # 트레인 데이터로 핏해서 엑스 테스트로 프래딕한거에다가 컬럼을 분위수의 값으로 넣음
    LGBM_actual_pred.columns=quantiles
    
    # 엑스 트레인 데이터로 핏한 모델이랑, 엑스 테스로 프래딕한 분위수 컬럼을 가지는 판다스 타입 데이터를 반환함.
    # 그리고 이제 파일을 저장하는 일만 남은 듯
    # 금방하시네 이분은...
    return LGBM_models, LGBM_actual_pred

# Target1
# 테스트 데이터 함수를 호출해서 위에 스플릿한 데이터_1 페밀리어를 넣고 반환 값으로 각각 초기화 함
# 엑스 값은  타겟 1, 2는 차이 없음 출력 값만 타겟 1에서는 1일 뒤 임
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
print(results_1.sort_index()[:48])

# Target2
# 얘는 2일 뒤의 핏한 모델과 예상치를 반환 값으로 받아 각각 초기화 함.
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)

# 2일 뒤의 예상값을 확인함 인덱스 기준으로 잘라 샘플을 봄
print(results_2.sort_index()[:48])

# 이건 1일 뒤의 예상값 확인함 이건 인덱스 로케이션을 찍엇네? (왜지??)
print(results_1.sort_index().iloc[:48])

# 2일 뒤 예상값의 처음과 끝을 같이 보려함
print(results_2.sort_index())

# 두 예상값의 크기를 비교함
print(results_1.shape, results_2.shape)

# 제출할 파일폼의 문자 로케이션에 반환된 1, 2일 뒤의 예상치의 값을 넣어 초기화 시킴
# 제출할 파일의 id 열에 문자형 Day7이 있는 행에 q_0.1로 시작하는 열부터 마지막 열까지 7일에 1일 뒤의 예상치의 값을 넣고 초기화 함.
# 아래는 Day8이 있는 행에 2일 뒤의 예상치를 넣고 초기화 함.
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

# 값이 잘 들어갔는지 처음과 끝부분을 확인
print(submission)

# 인덱스 넘버 로케이션으로 처음 48열에 값을 확인
print(submission.iloc[:48])

# 인덱스 넘버 로케이션으로 끝 48열에 값을 확인
print(submission.iloc[48:96])

# 만족하고 저장
submission.to_csv('../data/csv/submission_v5_lgbm.csv', index=False)

# submission_v5_lgbm_500147
# 이슬점, 일사량 함수 안쓴거
# 2.0202123047

# submission_v5_lgbm_500193
# 이슬점, 일사량 함수 쓴거
# 1.9973118521

# scaler = MinMaxScaler() 
# 분포표 출력시 0~1 사이값이라  크게 변하는거 없음
# 데이콘에 파일 밀어 넣어보고 뭐가 좋은지 보려함
# submission_v5_lgbm_500200
# 2.0045336305	


# scaler = StandardScaler() 
# 다음은 이거 오늘은 못할듯 제출 다함


# 그다음은 러닝레이트 조정하는 콜백 넣을거임
