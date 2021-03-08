# 이상치 처리
# 1. 0 처리
# 2. Nan 처리 후 보간
# 3.4.5.6... 알아서해

import numpy as np
aaa = np.array([1,2,3,4,6,7,90,100,300,200])

def outliers(data_out):
    # np.percentile 지정된 축을 따라 데이터의 q 번째 백분위 수를 계산.
    # 값을 순서대로 정렬하여 자리수의 퍼센트 값을 구하여 분위수로 받음
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 : ", quartile_1 )
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3 )
    # 25~75%를 기준대로
    iqr = quartile_3 - quartile_1
    # 1.5배 이상, 이하를 정상범위로 지정함
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    # 범위를 벗어난 값의 자리수를 반환함.
    return np.where((data_out>upper_bound)|(data_out<lower_bound))

outliers_loc = outliers(aaa)
print("이상치의 위치 :", outliers_loc)

# 실습
# 위 aaa 데이터를 boxplot으로 그리시오!!
import matplotlib.pyplot as plt
import pandas as pd
plt.figure(figsize=(7,6)) #크기 지정
aaas = pd.DataFrame(aaa, columns=['aaa'])
boxplot = aaas.boxplot(column=['aaa'])
plt.yticks(np.arange(0,101,step=1))
plt.show()