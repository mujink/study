# 데이터 크기를 맞춘뒤 저장한 CSV 파일을 로드 => 컬럼을 늘리지 않는 한, 다시 확인할 필요 없음.
# 데이터와 타겟 사이 상관계수 최종 확인. => 완료시 중간저장한 CSV 파일은 건드리지 않음.
# 전처리 미스로 로스값이 높음 => minmax 이전에 x 값, y 값 중간확인 필요.
# 확인은 CSV 파일이 편하므로 x,y 스플릿 후 각각 CSV 파일로 저장하는 작업을 실행함.
# 각각 작업은 CSV 파일로 변환하여 확인하고 주석처리로 하나하나 확인함.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops.np_array_ops import array

# 함수정의
# seq : 2차원 배열의 데이터 셋
# len(Seq) : 데이터 셋의 행의 길이
# x_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
# y_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.


# range(0, 1079)
# (1085, 6)
# 아래 주석을 (데이터의 길이 - 자르고 싶은 사이즈 -1) 회 만큼 반복함  
# seq의 데이터 셋에 [i열 부터  x_size 까지의 행, 첫번째 열부터 x_col 열까지]의 2차원 배열을 주소에 있는 값을 플롯으로 바꾸고 subset에 저장함
# seq의 데이터 셋에 [i + x_size 행, y_col 열]의 주소에 있는  1차원에 배열에 주소에 해당하는 값을 플롯으로 바꾸고 subset1에 저장함
# seq의 데이터 셋에 [i + x_size행의 다음 행, 해당 열]의 주소에 있는 값을 플롯으로 바꾸고 subset2에 저장함
# subset, subset1, subset2를 인수로 같는 np.array를 aaa라는 리스트의 다음 빈 행에 추가함.
# 위에서 len(seq)-x_size-1 회 만큼 반복하여 aaa에 쌓여있는 텐서를 함수를 불러온 곳으로 리턴함.

"""
split_xy 함수
아래 함수 검증 완료됨 
"""
def split_xy(seq,x_size,x_col,y_size,y_col):
    print(range(len(seq)-x_size-1))                             
    print(seq.shape)                                            
    aaa=[]
    for i in range(len(seq)-x_size-1):                          
        subset = seq[i:(i+x_size),0:x_col].astype('float32')   
        subset1 = seq[(i+y_size),y_col].astype('float32')       
        subset2 = seq[(i+y_size+1),y_col].astype('float32')     
        aaa.append(np.array([subset,subset1,subset2]))          
    print(np.array(aaa).shape)
    return np.array(aaa)                                        

# # 불러오기 ===============================================================================================

datasets = pd.read_csv('../data/csv/sam_dc.csv',index_col=0 ,encoding='ms949')

# len(Seq) : 데이터 셋의 행의 길이 현재 행의 길이는 1085임
# x_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
# y_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
#=========================================================================================================
x_size = 5
x_col = 6
y_size = x_size
y_col = 0

# DataFrame은 스플릿이 안됌. 왜인지 모르겟음.
datasets = datasets.to_numpy()
# x y데이터 생성, 스플릿
x1 = split_xy(datasets,x_size,x_col,y_size,y_col)
# 입력 데이터의 길이가 맞는지 확인
print("X1 :", x1.shape)     #(1079,5,6)

# 샘플 저장 및 확인=================================================================================================
# 넘파이는 CSV 저장이 안되서 데이터 프레임을 씌움, 2D 보다 높은 차원은 데이터프레임이 적용되지 않음.
# 출력 이상없음.

# +++++++++++++++++++++++++++++++++++
# x1 = (1079, 5, 6).astype('float32') 
# x2 = (1079, 5, 5).astype('float32') 
# y  = (1079, 2).astype('float32') 
# x1_prd.shape = (1,5,6)
# x2_prd.shape = (1,5,6)
# +++++++++++++++++++++++++++++++++++
x_STERT = pd.DataFrame(x1[1])
x_END = pd.DataFrame(x1[-1])
# 샘플 데이터는 확인용 데이터 이므로 로드하여 사용하지 않음.
x_STERT.to_csv('../data/csv/split_xy_STR.csv', encoding='ms949', sep=",")
x_END.to_csv('../data/csv/split_xy_END.csv', encoding='ms949', sep=",")
# 샘플 확인됨=================================================================================================
print(x_STERT)
print(x_STERT.shape)
print(x_END)
print(x_END.shape)


