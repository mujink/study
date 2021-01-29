#  feature_importances_
#  max_depth
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()
# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = RandomForestRegressor(max_depth=13)

# 3 fit
model.fit(x_train, y_train)

# 4 evel
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    plt.figure(figsize=(10,6))
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

"""
[0.04059332 0.00071107 0.00702888 0.00119282 0.02478404 0.40531769
 0.01598097 0.07159451 0.00233074 0.01457094 0.01600809 0.01034297
 0.38954394]
acc : 0.8898326188969536
"""