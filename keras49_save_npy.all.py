# boston, diabets, cancer, iris, wine
# mnist, fasion, cifar10, cifar100,
import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
#1. boston
boston_dataset = load_boston()
boston_x = boston_dataset.data
boston_y = boston_dataset.target
np.save('../data/npy/boston_x.npy', arr = boston_x)
np.save('../data/npy/boston_y.npy', arr = boston_y)

#2. diabet
diabet_dataset = load_diabetes()
diabet_x = diabet_dataset.data
diabet_y = diabet_dataset.target

np.save('../data/npy/diabet_x.npy', arr = diabet_x)
np.save('../data/npy/diabet_y.npy', arr = diabet_y)

#3. cancer
cancer_dataset = load_breast_cancer()
cancer_x = cancer_dataset.data
cancer_y = cancer_dataset.target

np.save('../data/npy/cancer_x.npy', arr = cancer_x)
np.save('../data/npy/cancer_y.npy', arr = cancer_y)

#4. iris
iris_dataset = load_iris()
iris_x = iris_dataset.data
iris_y = iris_dataset.target

np.save('../data/npy/iris_x.npy', arr = iris_x)
np.save('../data/npy/iris_y.npy', arr = iris_y)

#5. wine
wine_dataset = load_wine()
wine_x = wine_dataset.data
wine_y = wine_dataset.target

np.save('../data/npy/wine_x.npy', arr = wine_x)
np.save('../data/npy/wine_y.npy', arr = wine_y)

#6. mnist
(m_x_train, m_y_train), (m_x_test, m_y_test) = mnist.load_data()
np.save('../data/npy/mnist_x_train.npy', arr = m_x_train)
np.save('../data/npy/mnist_x_test.npy', arr = m_x_test)
np.save('../data/npy/mnist_y_train.npy', arr = m_y_train)
np.save('../data/npy/mnist_y_test.npy', arr = m_y_test)

#7. tashion
(t_x_train, t_y_train), (t_x_test, t_y_test) = fashion_mnist.load_data()
np.save('../data/npy/tahion_x_train.npy', arr = t_x_train)
np.save('../data/npy/tahion_x_test.npy', arr = t_x_test)
np.save('../data/npy/tahion_y_train.npy', arr = t_y_train)
np.save('../data/npy/tahion_y_test.npy', arr = t_y_test)

#8. cifer10
(cf10_x_train, cf10_y_train), (cf10_x_test, cf10_y_test) = cifar10.load_data()
np.save('../data/cifer10_x_train.npy', arr = cf10_x_train)
np.save('../data/cifer10_x_test.npy', arr = cf10_x_test)
np.save('../data/cifer10_y_train.npy', arr = cf10_y_train)
np.save('../data/cifer10_y_test.npy', arr = cf10_y_test)

#9. cifer100
(cf100_x_train, cf100_y_train), (cf100_x_test, cf100_y_test) = cifar100.load_data()
np.save('../data/npy/cifer100_x_train.npy', arr = cf100_x_train)
np.save('../data/npy/cifer100_x_test.npy', arr = cf100_x_test)
np.save('../data/npy/cifer100_y_train.npy', arr = cf100_y_train)
np.save('../data/npy/cifer100_y_test.npy', arr = cf100_y_test)
