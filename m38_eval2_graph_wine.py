from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import r2_score, accuracy_score


x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, shuffle=True, random_state = 1
)

# model
model = XGBClassifier(n_estimators=1000, learning_rate=0.01, n_jobs=8)

# fit
model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss','merror','cox-nloglik'],
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

epochs = len(result['validation_0']['cox-nloglik'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['cox-nloglik'], label='Train')
ax.plot(x_axis, result['validation_1']['cox-nloglik'], label='Test')
ax.legend()
plt.ylabel('cox-nloglik')
plt.title('XGBoost cox-nloglik')

# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['merror'], label='Train')
# ax.plot(x_axis, result['validation_1']['merror'], label='Test')
# ax.legend()
# plt.ylabel('merror')
# plt.title('XGBoost merror')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['merror'], label='Train')
ax.plot(x_axis, result['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')

plt.show()


"""
score : 0.9444444444444444
acc :  0.9444444444444444
================================================
result : {
    'validation_0': OrderedDict([('mlogloss', 
    [1.085054, 1.071726, 1.058622, 1.045738, 1.033066, 1.020603, 1.008343, 0.99628, 0.984412, 0.972733, 0.961237, 0.949931, 
    0.9388, 0.927841, 0.917049, 0.906421, 0.895954, 0.885645, 0.875547, 0.8656, 0.855798, 0.846139, 0.836612, 0.827231, 0.817985, 
    0.808872, 0.799887, 0.791021, 0.782288, 0.773677, 0.765178, 0.756805, 0.748547, 0.740402, 0.732361, 0.724436, 0.716619, 0.708899, 
    0.701291, 0.693776, 0.686369, 0.67906, 0.67184, 0.664722, 0.657697, 0.650756, 0.643913, 0.637151, 0.630483, 0.6239, 0.617395, 
    0.61095, 0.60458, 0.598296, 0.592085, 0.585958, 0.579901, 0.573925, 0.568018, 0.562188, 0.556426, 0.550739, 0.545117, 0.539568, 
    0.534081, 0.528666, 0.523312, 0.518027, 0.5128, 0.507641, 0.502539, 0.497502, 0.49252, 0.487598, 0.482737, 0.47793, 0.473183, 
    0.468485, 0.463849, 0.45927, 0.454744, 0.450264, 0.44581, 0.441443, 0.437091, 0.432793, 0.42854, 0.424339, 0.420184, 0.416078, 
    0.412017, 0.408005, 0.404035, 0.40011, 0.396232, 0.392395, 0.388604, 0.384853, 0.381147, 0.377479])]), 
    'validation_1': OrderedDict([('mlogloss', 
    [1.086543, 1.074682, 1.063025, 1.051565, 1.040298, 1.029218, 1.018322, 1.007605, 0.997063, 0.98669, 0.976484, 0.966438, 
    0.956465, 0.946647, 0.937201, 0.9279, 0.918526, 0.909294, 0.900007, 0.890987, 0.881898, 0.873023, 0.864281, 0.855588, 0.847101, 
    0.838659, 0.83053, 0.822409, 0.814513, 0.806542, 0.798765, 0.791097, 0.783538, 0.776006, 0.768659, 0.761511, 0.754366, 0.74732, 
    0.740465, 0.733612, 0.726772, 0.720024, 0.713447, 0.706878, 0.700562, 0.694246, 0.688015, 0.681867, 0.67572, 0.669814, 0.663906, 
    0.658061, 0.652293, 0.6466, 0.640982, 0.635512, 0.630038, 0.624554, 0.619221, 0.613874, 0.608677, 0.603615, 0.598549, 0.593547, 
    0.588609, 0.583733, 0.578919, 0.57423, 0.569536, 0.564818, 0.560241, 0.555638, 0.551173, 0.546765, 0.542469, 0.538169, 0.533899, 
    0.529506, 0.52534, 0.521232, 0.517192, 0.513035, 0.50887, 0.504851, 0.500782, 0.496798, 0.492823, 0.488788, 0.484905, 0.48105, 
    0.477255, 0.473482, 0.469774, 0.466093, 0.462386, 0.458783, 0.455286, 0.451763, 0.44837, 0.444931])])
    }
"""