#  feature_importances_
#  max_depth

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2 model
model = DecisionTreeClassifier(max_depth=4)

# 3 fit
model.fit(x_train, y_train)

# 4 evel
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)
"""
[0.         0.00787229 0.96203388 0.03009382]
acc : 0.9333333333333333
"""