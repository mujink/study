#  이진 맹그러
#  skleaen에 Accuracy 쓸 것
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

fd = {x:x_data ,y:y_data}

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))

# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-2).minimize(cost)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0011).minimize(cost)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00073).minimize(cost)


# cast => hypothesis > 0.5 가 True 면 1, False 면 0 반환
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# predicted와 y가 같은지 비교하고 참이면 1, 거짓이면 0을 출력해서 평균을 구함
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(4200):
        cost_val, _ = sess.run([cost, train], feed_dict=fd)
        if step % 200 ==0:
            print(step,"cost :", cost_val)
    h, p ,a = sess.run([hypothesis, predicted, accuracy], feed_dict=fd)
    acc = accuracy_score(p, y_data)
    print("n Accuracy :", a, "\n acc :", acc)

# n Accuracy : 0.93497366 
#  acc : 0.9349736379613357