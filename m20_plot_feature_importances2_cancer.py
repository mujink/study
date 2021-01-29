#  feature_importances_
#  max_depth

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = DecisionTreeClassifier(max_depth=30)

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
[0.         0.         0.         0.         0.         0.
 0.00621763 0.         0.         0.         0.         0.
 0.         0.00299321 0.         0.0186973  0.         0.00621763
 0.00621763 0.         0.         0.05204668 0.715762   0.00925886
 0.04409158 0.         0.         0.13849747 0.         0.        ]
acc : 0.9385964912280702
"""