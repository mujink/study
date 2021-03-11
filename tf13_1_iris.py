import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

tf.set_random_seed(66)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])

dataset = load_iris()

x_data = dataset.data
y_datas = dataset.target
# (150, 4)
# (150,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_trains, y_tests = train_test_split(x_data, y_datas, train_size = 0.8, shuffle = True, random_state=1)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_trains.reshape(-1,1)
y_test = y_tests.reshape(-1,1)
one.fit(y_train)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
# (178, 13)
# (178, 3)

fd = {x:x_train, y:y_train}
fdt = {x:x_test, y:y_test}

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,3]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

hy = tf.argmax(y,1)
predicted = tf.argmax(hypothesis,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,hy), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(4200):
        loss_val, p, Acc,_ = sess.run([loss, predicted, accuracy, train], feed_dict=fd)
        if step % 200 ==0:
            y_tr = sess.run(tf.argmax(y_train,1))
            acc = accuracy_score(p, y_tr)
            print(step,"cost :", loss_val, "acc :", acc, "Acc :", Acc)
    loss_val, p, _ = sess.run([loss, hypothesis, train], feed_dict=fdt)
    ps = sess.run(tf.argmax(p,1))
    y_te = sess.run(tf.argmax(y_test,1))
    acc = accuracy_score(ps, y_te)
    print("test cost :", loss_val, "test acc :", acc)
