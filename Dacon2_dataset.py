import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/csv/Dacon2/data/train.csv', index_col=0, header = 0)

print(dataset)
"""
print(dataset.iloc[0,:])

digit     5
letter    L
0         1
1         1
2         1
         ..
779       4
780       4
781       4
782       3
783       4
"""
x = dataset.iloc[:,2:]
y = dataset.iloc[:,2]
noise_letter = dataset.iloc[:,1]
print(x.head())
# [2048 rows x 786 columns]
#     0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  ...  766  767  768  769  770  771  772  773  774  775  776  777  778  779  780  781  782  783
# id                                                                ...
# 1   1  1  1  4  3  0  0  4  4  3   0   4   3   3   3   4   4   0  ...    0    4    2    0    3    4    1    1    2    1    0    1    2    4    4    4    3    4 
# 2   0  4  0  0  4  1  1  1  4  2   0   3   4   0   0   2   3   4  ...    4    2    2    4    4    0    4    2    0    3    0    1    4    1    4    2    1    2 
# 3   1  1  2  2  1  1  1  0  2  1   3   2   2   2   4   1   1   4  ...    0    3    2    0    2    3    0    2    3    3    3    0    2    0    3    0    2    2 
# 4   1  2  0  2  0  4  0  3  4  3   1   0   3   2   2   0   3   4  ...    2    4    1    4    0    1    0    4    3    3    2    0    1    4    0    0    1    1 
# 5   3  0  2  4  0  3  0  4  2  4   2   1   4   1   1   4   4   0  ...    2    2    1    4    2    1    2    1    4    4    3    2    1    3    4    3    1    2 
print(y.head())
# [5 rows x 784 columns]
#     digit
# id
# 1       5
# 2       0
# 3       4
# 4       9
# 5       6
print(noise_letter.head())
# id
# 1    L
# 2    B
# 3    L
# 4    D
# 5    A