
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops.np_array_ops import array

# 함수정의
# x, and y 스플릿 함수. 각각 생성하여 따로 반한하게 만듬.
# len(Seq) : 데이터 셋의 행의 길이 현재 행의 길이는 1085임
# size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
def split_x(seq,size,col):
    print(range(len(seq)-size-1))
    print(seq.shape)
    aaa=[]
    for i in range(len(seq)-size-1):                     
        subset = seq[i:(i+size),0:col].astype('float32') 
        aaa.append(np.array(subset))
    print(np.array(aaa).shape)
    return np.array(aaa)

def split_y(seq,size,col):
    print(range(len(seq)-size-1))
    print(seq.shape)
    aaa=[]
    for i in range(len(seq)-size-1):
        subset = seq[(i+size),col].astype('float32')
        subset1 = seq[(i+size+1),col].astype('float32')
        aaa.append(np.array([subset,subset1]))
        # aaa.append(np.array())
    print(np.array(aaa).shape)
    return np.array(aaa)
# # 불러오기 ===============================================================================================

datasets = pd.read_csv('../data/csv/sam_dc.csv',index_col=0 ,encoding='ms949')
codacdbset = pd.read_csv('../data/csv/codac150_dc.csv',index_col=0, encoding='ms949')


# 몇일 분량, 컬럼확인
# 컬럼과 사이즈는 데이터 컬럼, 데이터의 수량 보다 많을 수 없음.
size=5
sam_col=6
cdc_col=5

datasets.to_numpy()
codacdbset.to_numpy()
# x1은 삼성, x2는 코스닥, y는 삼성.
x1 = datasets.to_numpy()
x2 = codacdbset.to_numpy()
y = datasets.to_numpy()


# x y데이터 생성, 스플릿
x1 = split_x(x1,size,sam_col)
x2 = split_x(x2,size,cdc_col)
y = split_y(y,size,0)

# 샘플 저장 및 확인=================================================================================================
# 넘파이는 CSV 저장이 안되서 데이터 프레임을 씌움, 2D 보다 높은 차원은 데이터프레임이 적용되지 않음.
# 출력 이상없음.
# @@출력의 형태는 2차원을 넘어갈시 []리스트를 포함하여 값으로 저장됨.
# +++++++++++++++++++++++++++++++++++
# x1 = (1079, 5, 6).astype('float32') 
# x2 = (1079, 5, 5).astype('float32') 
# y  = (1079, 2).astype('float32') 
# x1_prd.shape = (1,5,6)
# x2_prd.shape = (1,5,6)
# +++++++++++++++++++++++++++++++++++
x1 = pd.DataFrame(x1[-1])
x2 = pd.DataFrame(x2[-1])
y = pd.DataFrame(y)
# 샘플 데이터는 확인용 데이터 이므로 로드하여 사용하지 않음.
x1.to_csv('./csv/x1.csv', encoding='ms949', sep=",")
x2.to_csv('./csv/x2.csv', encoding='ms949', sep=",")
y.to_csv('./csv/y.csv', encoding='ms949', sep=",")
# 샘플 확인됨=================================================================================================
