import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
# y = dataset['target']
y = dataset.target

# df = pd.DataFrame(x, columns=dataset['feature_names'])
df = pd.DataFrame(x, columns=dataset.feature_names)

df['Target'] = y

# csv만들기
df.to_csv('../data/csv/iris_sklearn.csv', sep=',')