from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import r2_score, accuracy_score


x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, shuffle=True, random_state = 1
)

# model
model = XGBClassifier(n_estimators=1000, learning_rate=0.01, n_jobs=8)

# fit
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss','error','cox-nloglik'],
        eval_set=[(x_train,y_train),(x_test,y_test)],
        early_stopping_rounds=10
            )

aaa = model.score(x_test,y_test)
print("score :", aaa)

y_prad = model.predict(x_test)
acc = accuracy_score(y_test, y_prad)
print("acc : ", acc)

print("================================================")

result = model.evals_result()
# 발리데이션 로스 값을 출력
print("result :", result)



import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['cox-nloglik'], label='Train')
ax.plot(x_axis, result['validation_1']['cox-nloglik'], label='Test')
ax.legend()
plt.ylabel('cox-nloglik')
plt.title('XGBoost cox-nloglik')

# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel('logloss')
# plt.title('XGBoost logloss')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['error'], label='Train')
ax.plot(x_axis, result['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')

plt.show()

"""
score : 0.9473684210526315
acc :  0.9473684210526315
================================================
result : {'validation_0': OrderedDict([('logloss', 
        [0.684032, 0.675133, 0.666401, 0.657832, 0.649421, 0.641164, 0.633057, 0.625095, 0.617273, 0.60959, 0.602041, 0.594622, 
        0.587331, 0.580165, 0.57312, 0.566193, 0.559382, 0.552684, 0.546096, 0.539615, 0.53324, 0.526968, 0.520796, 0.514722, 
        0.508745, 0.502862, 0.49707, 0.491369, 0.485756, 0.480228, 0.474785, 0.469425, 0.464145, 0.458945, 0.453823, 0.448776,
         0.443804, 0.438905, 0.434077, 0.42932, 0.424632, 0.420011, 0.415457, 0.410967, 0.406628, 0.402263, 0.397959, 0.393799, 
         0.389613, 0.385567, 0.381494, 0.377398, 0.373516, 0.36953, 0.365672, 0.361792, 0.357965, 0.354309, 0.350582, 0.346952, 
         0.343323, 0.339788, 0.336253, 0.332812, 0.329368, 0.326015, 0.32266, 0.319394, 0.316124, 0.312941, 0.309774, 0.306672, 
         0.303564, 0.30054, 0.29751, 0.294562, 0.291626, 0.288751, 0.285914, 0.28307, 0.280302, 0.277529, 0.274831, 0.272124, 
         0.269492, 0.266844, 0.264229, 0.261692, 0.259139, 0.256663, 0.254171, 0.251756, 0.249322, 0.246965, 0.244605, 0.242303,
         0.24001, 0.237762, 0.235535, 0.233292])]), 
         'validation_1': OrderedDict([('logloss', 
         [0.685107, 0.677181, 0.66941, 0.661726, 0.654188, 0.646863, 0.63975, 0.632613, 0.62565, 0.618969, 0.612154, 0.605712, 
         0.599237, 0.592831, 0.586725, 0.580635, 0.574544, 0.568698, 0.56287, 0.557272, 0.551538, 0.546086, 0.540602, 0.535362, 
         0.530057, 0.524997, 0.519801, 0.514894, 0.510037, 0.505147, 0.500309, 0.495686, 0.491132, 0.48653, 0.482158, 0.477711, 
         0.473411, 0.469086, 0.464876, 0.460799, 0.45672, 0.452666, 0.448807, 0.444858, 0.441219, 0.437498, 0.433705, 0.430088, 
         0.426559, 0.423063, 0.419539, 0.415671, 0.412427, 0.408602, 0.40537, 0.401657, 0.398039, 0.394841, 0.391327, 0.388134, 
         0.384654, 0.381549, 0.378211, 0.375191, 0.37189, 0.368953, 0.365787, 0.362929, 0.359796, 0.357016, 0.354069, 0.351363, 
         0.348434, 0.3458, 0.342954, 0.34039, 0.337662, 0.335166, 0.332703, 0.330191, 0.327792, 0.325344, 0.323174, 0.320788, 
         0.318729, 0.31619, 0.31369, 0.311638, 0.309202, 0.307345, 0.304968, 0.303049, 0.30078, 0.298835, 0.296653, 0.294855, 
         0.292887, 0.2911, 0.289283, 0.287145])])
         }  
"""