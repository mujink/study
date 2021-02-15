from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import r2_score, accuracy_score


x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, shuffle=True, random_state = 1
)

# model
model = XGBRegressor(n_estimators=10, learning_rate=0.01, n_jobs=8)

# fit
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',
        eval_set=[(x_train,y_train),(x_test,y_test)]
            )

aaa = model.score(x_test,y_test)
print("score :", aaa)

y_prad = model.predict(x_test)
r2 = r2_score(y_test, y_prad)
print("r2 : ", r2)

print("================================================")

result = model.evals_result()
# 발리데이션 로스 값을 출력
print("result :", result)
"""
score : 0.9159675084902162
r2 :  0.9159675084902162
"""