import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="weight")
print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


# 변수 실행 방법들 3가지
# Session() 지정 후
# sess.run 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

aaa = sess.run(W)
print("aaa :", aaa)
sess.close()

# 세션을 InteractiveSession로 지정 한 다음에
# 베리어블.eval()을 사용하는 방법
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

bbb = W.eval()
print("bbb :",bbb)
sess.close()

# 세션을 사용하고
# 베리어블.eval()에 세션을 명시하는 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

ccc = W.eval(session=sess)
print("ccc :",ccc)
sess.close()
