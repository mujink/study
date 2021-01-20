# 태양광 발전량 예측하기.
# 데이터 셋을 테스트로 분류할 것
# 테스트 데이터는 프레딕트 할 데이터
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
================================================================================================================
train.csv : 훈련용 데이터 (1개 파일)
- 3년(Day 0~ Day1094) 동안의 기상 데이터, 발전량(TARGET) 데이터
================================================================================================================
shape => ( Day = 1094(1095), Hour : 23(24), Minute : 1(2), Parameter : 5, Target : 1 )
==============================================================###################################################
컬럼 설명                                                       파라미터     train 
# Index : 3
# Index Length (52560 => 1095*24*2)
Day - 일                                                                   0 ~ 1094
Hour - 시간                                                                0 ~ 23     
Minute - 분                                                                0 또는 30
# Parameter : 5
DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))   (W/m2)     0 ~ 528
DNI - 직달일사량(Direct Normal Irradiance (W/m2))               (W/m2)     0 ~ 1569    
WS - 풍속(Wind Speed (m/s))                                     (m/s)      0.0 ~ 12.0
RH - 상대습도(Relative Humidity (%))                            (%)        0.00 ~ 100.00
T - 기온(Temperature (Degree C))                                (C)        -19 ~ 35
# Target : 1
Target - 태양광 발전량 (kW)                                     (kW)       0.0 ~ 100+e8
==============================================================###################################################
"""

# File = 81
# TestDbSet =[]

# # 널 값을 뺴줌.
# def preprocess_data(data):
#     temp = data.copy()
#     return temp.iloc[-48:,:]

# #  81개 파일을 데이터 셋에 담음
# for i in range(File) :
#       file_path = '../data/csv/Dacon/test/' + str(i) + '.csv'
#       temp = pd.read_csv(file_path,encoding='ms949')
#       temp = preprocess_data(temp)
#       TestDbSet.append(temp)
# """
# test.csv : 정답용 데이터 (81개 파일)
# - 2년 동안의 기상 데이터, 발전량(TARGET) 데이터 제공 
# - 각 파일(*.csv)은 7일(Day 0~ Day6) 동안의 기상 데이터, 발전량(TARGET) 데이터로 구성
# - 파일명 예시: 0.csv, 1.csv, 2.csv, …, 79.csv, 80.csv (순서는 랜덤이므로, 시계열 순서와 무관)
# - 각 파일의 7일(Day 0~ Day6) 동안의 데이터 전체 혹은 일부를 인풋으로 사용하여, 향후 2일(Day7 ~ Day8) 동안의 30분 간격의 발전량(TARGET)을 예측 (1일당 48개씩 총 96개 타임스텝에 대한 예측)

# sample_submission.csv : 정답제출 파일
# - test 폴더의 각 파일에 대하여, 시간대별 발전량을 9개의 Quantile(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)에 맞춰 예측
# - “파일명_날짜_시간” 형식(예시: 0.csv_Day7_0h00m ⇒ 0.csv 파일의 7일차 0시00분 예측 값)에 유의
# """
# # 데이터 구조 확인.
# # 데이터 


# =========================================================================================================================================
# 이슬점 온도 및 일반 일사량을 계산하는 함수.
# 온도와 습도에 따라 조사량이 달라지는 문제 때문에 다음 함수를 사용함. (예: 습도 60도 이상에 이슬점 온도에 해당하면 안개나 비가내리거나 구름이 끼는 등)

"""
공기가 냉각되어 응결이 시작될 때의 온도
포화 상태에 도달할 때의 온도
상대 습도가 100%일 때의 온도
"""
# 온도가 올라가면 상대습도가 낮아지는 경향이 있음.
# 이건 따듯한 물에 커피가루가 잘 섞이는 것과 같음.
# 매질에 무언가 녹아서 섞이는 정도가 다름.
# 포화수증기량에서 곡선이 의미하는 바는 포화 곡선 이상의 수중기량은 응결된다라는 것임.
# 해서 습도 100%라고 하는 것은 온도에 따라 그 절대값이 변하는 것임.
# 이걸 포화수증기량에서 곡선을 통해 확인가능.
# 아래 함수는 이슬점을 계산하는 것이고 이슬점 온도 이하의 물체 근처에서 포화된 수중기량을 가진 기체는 이슬점에서 가질 수 있는 포화 수중기
# 를 넘어선 수중기가 응결된다.

"""
b = 17.62, c = 243.12℃ 인경우 -45℃ ≤ T ≤ 60℃ (오차율 ±0.35℃)
b = 17.27, c = 237.7℃ 인 경우 0℃ ≤ T ≤ 60℃ (오차율 ±0.4℃)
b = 17.368, c = 238.88℃ 인 경우 0℃ ≤ T ≤ 50℃ (오차율 ≤ 0.05%)
b = 17.966, c = 247.15℃ 인 경우 -40℃ ≤ T ≤ 0℃ (오차율 ≤ 0.06%)

Td = 이슬점 온도
T-Td = 공기 온도와 이슬점 온도차
GHI = 일반 일사량
"""
# 참고 링크 : https://makerjeju.tistory.com/24
# np.log => 괄호안에 값을 자연상수 로그한다.
def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']+data['DHI'])
    return data
# ====== 파일 로드 ================================================================================================================================

TrainDbSet = pd.read_csv('../data/csv/Dacon/preprocess_csv/TrainDbSet.csv',encoding='ms949', index_col=0)
TestDbSet = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet.csv',encoding='ms949', index_col=0)


TrainDbSet = Add_features(TrainDbSet)
TestDbSet = Add_features(TestDbSet)

# 드롭 안한게 더 잘나왔음 2
# del TrainDbSet['Hour']
# del TrainDbSet['Td']
# del TrainDbSet['WS']
# del TrainDbSet['RH']

# del TestDbSet['Hour']
# del TestDbSet['Td']
# del TestDbSet['WS']
# del TestDbSet['RH']

TrainDbSet.to_numpy()
print(TrainDbSet.shape)
print(TrainDbSet)

TrainDbSet.to_csv('../data/csv/Dacon/preprocess_csv/TrainDbSet2.csv', encoding='ms949', sep=",")
TestDbSet.to_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', sep=",")
# codacdbset.to_csv('../data/csv/codac150_dc.csv', encoding='ms949', sep=",")

TrainDbSet.to_numpy()
print(TrainDbSet.shape)

print(TrainDbSet.corr())
sns.set(font_scale=0.7)
sns.heatmap(data=TrainDbSet.corr(), square=True, annot=True, cbar=True)
plt.show()

print(TestDbSet.corr())
sns.set(font_scale=0.7)
sns.heatmap(data=TestDbSet.corr(), square=True, annot=True, cbar=True)
plt.show()
# print(TrainSet2.corr())
# sns.set(font_scale=0.7)
# sns.heatmap(data=TrainSet2.corr(), square=True, annot=True, cbar=True)
# plt.show()

# scaler = MinMaxScaler()
# scaler.fit(codacdbset)
# scaler.fit(datasets)
# datasets_minmaxed = scaler.fit_transform(datasets)
# codacdbset_minmaxed = scaler.fit_transform(codacdbset)