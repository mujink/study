# 데이터 자료 크기 맞춤
# 사용하지 않을 컬럼 제거 및 CSV 중간저장

import numpy as np
import pandas as pd

# 함수정의

def str_to_float(input_str):
    temp = input_str
    if temp[0]!='-':
        temp = input_str.split(',')
        sum = 0
        for i in range(len(temp)):
            sum+=float(temp[-i-1])*(10**(i*3))
        return sum
    else:
        temp=temp[1:]
        temp = input_str.split(',')
        sum = 0
        for i in range(len(temp)):
            sum+=float(temp[-i-1])*(10**(i*3))
        return -sum  

datasets = pd.read_csv("samsung.csv",encoding='cp949',index_col=0)
codacdbset = pd.read_csv("KODEX 코스닥150 선물인버스.csv",encoding='cp949',index_col=0)
datasets1 = pd.read_csv("삼성전자0115.csv",encoding='cp949',index_col=0)

#전처리 
#1-1 결측치 제거
datasets_1 = datasets.iloc[:662,:]
datasets_2 = datasets.iloc[665:,:]


datasets = pd.concat([datasets_1,datasets_2])

#1-1 앙상블 일자 겹치지 않는 행 제거
# 두 인풋과의 출력의 영향을 확인 할 수 없음, 그래고 데이터 길이를 맞춰 주기 위함.
datasets = datasets.iloc[:1083,:]
codacdbset_1 = codacdbset.iloc[:664,:]
codacdbset_2 = codacdbset.iloc[667:,:]
codacdbset = pd.concat([codacdbset_1,codacdbset_2])

# null 행렬 제거
datasets1 =  datasets1.iloc[:3,:]
codacdbset = codacdbset.iloc[:1315,:]

# 불필요한 행 제거
datasets =  datasets.drop(['2021-01-13'])

# 열 제거
del datasets1["전일비"]
del datasets1["Unnamed: 6"]
del codacdbset["전일비"]
del codacdbset["Unnamed: 6"]

# 합치기
datasets = pd.concat([datasets1,datasets])

# 결측 확인 및 타입변환
# str -> florat        
for j in [0,1,2,3,5,6,8,9,10,11,12]:
    for i in range(len(datasets.iloc[:,j])):
        datasets.iloc[i,j] = str_to_float(datasets.iloc[i,j])
    print(datasets.iloc[:0,j], j)

for j in [0,1,2,3,5,6,8,9,10,11]:
    for i in range(len(codacdbset.iloc[:,j])):
        codacdbset.iloc[i,j] = str_to_float(codacdbset.iloc[i,j])
    print(codacdbset.iloc[:0,j], j)

"""
print(datasets.isnull().sum())

시가        0
고가        0
저가        0
종가        0
등락률       0
거래량       0
금액(백만)    0
신용비       0
개인        0
기관        0
외인(수량)    0
외국계       0
프로그램      0
외인비       0
dtype: int64
"""
# # 50으로 나누고 곱함

# 내림차순
datasets = datasets.iloc[::-1,:]
codacdbset = codacdbset.iloc[::-1,:]
datasets.iloc[:421,0:4] = datasets.iloc[:421,0:4]/50.0
datasets.iloc[:421,5] = datasets.iloc[:421,5]*50

# 안쓸 컬럼 제거.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 상관계수에서 0에 수렴하는 컬럼은 좋지 않음. -1 또는 1에 가까운 경우가 좋음.

# del datasets["시가"]
# del datasets["고가"]
del datasets["저가"]
del datasets["종가"]
del datasets["등락률"]
# del datasets["거래량"]
# del datasets["금액(백만)"]
del datasets["신용비"]
# del datasets["개인"]
del datasets["기관"]
del datasets["외인(수량)"]
del datasets["외국계"]
# del datasets["프로그램"]
del datasets["외인비"]

del codacdbset["시가"]
# del codacdbset["고가"]
del codacdbset["저가"]
# del codacdbset["종가"]
del codacdbset["등락률"]
# del codacdbset["거래량"]
# del codacdbset["금액(백만)"]
del codacdbset["신용비"]
del codacdbset["개인"]
del codacdbset["기관"]
# del codacdbset["외인(수량)"]
del codacdbset["외국계"]
del codacdbset["프로그램"]
del codacdbset["외인비"]
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# 중간 저장=================================================================================================
datasets.to_csv('./csv/sam_dc.csv', encoding='ms949', sep=",")
codacdbset.to_csv('./csv/codac150_dc.csv', encoding='ms949', sep=",")
