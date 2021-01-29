#  feature_importances_
#  max_depth
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

dataset = load_diabetes()
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
[0.06908094 0.0093772  0.25943039 0.07759471 0.04385666 0.06431869
 0.05404213 0.02290888 0.33427698 0.06511342]
acc : 0.42369079218479866
acc : 0.8898326188969536
"""