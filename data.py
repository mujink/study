import pandas as pd

a = pd.read_csv("./data/dirty_mnist_2nd_answer.csv")

for index in a.columns :
    print(index)
    print(a[index].value_counts(normalize=True))