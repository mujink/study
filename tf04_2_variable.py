import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name="test")

# 변수 초기화 = > 텐서에서 쓸 수 있도록 초기화 해줌
# init = tf.global_variables_initializer()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print(sess.run(init)) #None
print(sess.run(x))