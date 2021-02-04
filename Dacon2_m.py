import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.decomposition import PCA

# 불러오기
train = pd.read_csv('../data/csv/Dacon2/data/train.csv')
test = pd.read_csv('../data/csv/Dacon2/data/test.csv')
# ==============================그림보기==========================
img = []
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
# ==============================데이터 전처리==========================

# 트래인의 아이디, 디지트, 레터 컬럼을 제거한 나머지 컬럼의 값을 축변경하여 초기화


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( 
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        # zoom_range=0.15,
        rotation_range = 10)
        # validation_split=0.2)
    
def imgGenerator(img, label):

    x_set, y_set = [],[]
    for i in range(img.shape[0]): #1024
        num_aug = 0
        x = img[i]  #1024
        y = label[i]  #1024
        x_t = x.reshape((1,) + x.shape)
        # b = np.array(x)
        # print(b.shape)
        # b = b.reshape(28,28)
        # plt.imshow(b)
        # plt.show() 
        for x_aug in datagen.flow(x_t) : #20
            if num_aug >= 50:
                break
            x_set.append(x_aug) #1024*20
            y_set.append(y)
            # a = np.array(x_aug)
            # print(a.shape)
            # a = a.reshape(28,28)
            # plt.imshow(a)
            # plt.show()
            # print(y)
            num_aug += 1

 
    
    x_set = np.array(x_set)
    y_set = np.array(y_set)


    x_set = x_set.reshape(-1, 28*28)
    y_set = y_set.reshape(-1)

    return x_set, y_set


#  모델을 정의하고 변수 트래인을 인자로 받음
# parameters = [
#     {"n_estimators" : [100,200,300], "learning_rate" : [0.1,0.3,0.001,0.01],
#     "max_depth":[4,5,6]},
#     {"n_estimators" : [90,100,110], "learning_rate" : [0.1,0.001,0.01],
#     "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
#     {"n_estimators" : [90,110], "learning_rate" : [0.1,0.001,0.5],
#     "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],
#     "colsample_bylevel":[0.6,0.7,0.9]},
# ]

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = np.where((x_train<=150)&(x_train!=0) ,0.,x_train)
y_train = train['digit'].values

x_test = test.drop(['id', 'letter'],axis=1).values
x_test = np.where((x_test<=150)&(x_test!=0) ,0.,x_test)
x_test = x_test.reshape(-1,28*28)


x_train = x_train.reshape(-1,28,28,1)



x_train, y_train = imgGenerator(x_train, y_train)

x_train = x_train/255
x_test = x_test/255

pca = PCA()
x2 = pca.fit_transform(x_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.99)+1

pca = PCA(d+10)
pca.fit(x_train)

from sklearn.model_selection import train_test_split
x_train, x_test1,  y_train, y_test1 = train_test_split(x_train, y_train, shuffle = True, random_state=1, test_size=0.2)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)

for train_index, valid_index in skf.split(x_train,y_train) :
    ind = 0
    x_train1, x_val1 = x_train[train_index], x_train[valid_index]
    y_train1, y_val1 = y_train[train_index], y_train[valid_index]
    print(y_train1)
    print(y_train1.shape)
    print(x_train1.shape)
    print(x_val1.shape)

    # model = XGBClassifier()

    # model.load_model('../data/xgb_save/Dacon2_m.model')

    model = XGBClassifier(n_estimators=200, n_jobs=8,
                        learning_rate=0.1, max_depth= 100,
                         colsample_bytree=1, colsample_bylevel=1)
    # model = XGBClassifier()
    # model.load_model('../data/xgb_save/Dacon2_m.model')
    result = model.fit(x_train1, y_train1, verbose=1, eval_metric=['mlogloss','merror','cox-nloglik'],
            eval_set=[(x_val1, y_val1)])
    # result = model.fit(x_train1, y_train1, verbose=1, eval_metric=['logloss','error'], early_stopping_rounds=3,
    #     eval_set=[(x_val1, y_val1)])
    # plot_importance(model)
    # plt.show()

    model.save_model('../data/xgb_save/Dacon2_'+str(ind)+'m.model')

    r2 = model.score(x_test1,y_test1)
    r2set = []
    r2set.append(r2)
    print("r2 :", r2)

    del x_train1
    del y_train1
    del x_val1
    del y_val1

print ("r2set :",r2set)
# 발리데이션 로스 값을 출력

# import matplotlib.pyplot as plt

# epochs = len(result['validation_0']['mlogloss'])
# x_axis = range(0,epochs)


# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['merror'], label='val')
# ax.legend()
# plt.ylabel('merror')
# plt.title('XGBoost merror')

# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['mlogloss'], label='val')
# ax.legend()
# plt.ylabel('mlogloss')
# plt.title('XGBoost mlogloss')

# plt.show()
# ============================================


submission = pd.read_csv('../data/csv/Dacon2/data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test))
print(submission.head())

submission.to_csv('../data/csv/Dacon2/data/farst_xgm_dacon2.csv', index=False)
