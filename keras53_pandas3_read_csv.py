import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header = 0)
"""
index_col = 0 : 인덱스 0 은 인덱스가 아님.
header = 0 : 인덱스 0은 헤더임.
"""
print(df)