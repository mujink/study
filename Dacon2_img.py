import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from xgboost import XGBClassifier

# 불러오기
train = pd.read_csv('../data/csv/Dacon2/data/train.csv',header = 0)
test = pd.read_csv('../data/csv/Dacon2/data/test.csv',header = 0)
# ==============================그림보기==========================
idx = 14
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train.loc[idx, 'digit']
letter = train.loc[idx, 'letter']

plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
plt.show()