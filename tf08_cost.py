import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [2.,4.,6.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x*w

cost = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess:
    # sess.run(tf.compat.v1.global_variables_initializer()) #텐서플로의 베리어블을 초기화함 
    for i in range(-30,50):
        curr_w = i * 0.1
        curr_cost  = sess.run(cost, feed_dict={w:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("==========================================")
print(w_history)
print("==========================================")
print(cost_history)
print("==========================================")

plt.plot(w_history, cost_history)
plt.show()

# 웨이트를 넣었을 때 cost 함수 가 변하는 과정을 출력함