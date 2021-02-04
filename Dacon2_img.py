import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from xgboost import XGBClassifier

train = pd.read_csv('../data/csv/Dacon2/data/train.csv',header = 0)
test = pd.read_csv('../data/csv/Dacon2/data/test.csv',header = 0)
# ==============================그림보기==========================
idx = 2000
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train.loc[idx, 'digit']
letter = train.loc[idx, 'letter']

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)

plt.title('img1 Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)

plt.subplot(1,2,2)

img2 = np.where((img<=150)&(img!=0) ,0.,img)
plt.title('img2 Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img2)
plt.show()