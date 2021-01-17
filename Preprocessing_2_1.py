#  출력에 문제가 있음. 다른 해결이 되지 않음.
#  R2 값을 구하기 위해 x_test에 대한 y_predict 출력시, 3차원 배열이 출력되어
#  비교하는 두 변수 어레이 배열이 달라 R2가 출력되지 않는 문제임
#  입력 배열 => 216,5,6 / 216,5,5
#  원하는 배열 => 216,2임
#  출력되는 배열 => 216, 5, 2
#  스플릿 X,Y 시에 일자별로 출력되는 것은 문제가 되지 않을 것으로 봄
#  transpose 를 적용해야 할 듯 함.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# X1 : (1079, 5, 6)
# X2 : (1079, 5, 5)
# Y: (1079, 2)

# X1 Train : (690, 5, 6) X2 Trian : (690, 5, 5)
# Y1 Train : (690, 2) <class 'numpy.ndarray'>
# X1 Tset  : (216, 5, 6)  X2 Test  : (216, 5, 5)
# Y1 Tset  : (216, 2) <class 'numpy.ndarray'>
# X1 Val   : (173, 5, 6)   X2 val   : (173, 5, 5)
# Y1 Val   : (173, 2) <class 'numpy.ndarray'>
# X1_prd   : (1, 5, 6)   X2_prd   : (1, 5, 5)

# # #  넘파이 불러오기=======================================================


x1_train = np.load('./npy/1.npy',allow_pickle=True)
x2_train = np.load('./npy/2.npy',allow_pickle=True)
y_train = np.load('./npy/3.npy',allow_pickle=True)
x1_test = np.load('./npy/11.npy',allow_pickle=True)
x2_test = np.load('./npy/12.npy',allow_pickle=True)
y_test = np.load('./npy/13.npy',allow_pickle=True)
x1_val = np.load('./npy/21.npy',allow_pickle=True)
x2_val = np.load('./npy/22.npy',allow_pickle=True)
y_val = np.load('./npy/23.npy',allow_pickle=True)
x1_prd = np.load('./npy/31.npy',allow_pickle=True)
x2_prd = np.load('./npy/32.npy',allow_pickle=True)

x1_train = np.transpose(x1_train)
print(x1_train.shape)