import tensorflow as tf
x = [1,2,3]
W = tf.compat.v1.Variable([0.3], tf.float32)
B = tf.compat.v1.Variable([1.0], tf.float32)

hypothesis = x * W + B
print("hypothesis :",hypothesis)
# 실습
# 1. sess.run()
# 2. InteractiveSession
# 3. .eval(session=sess)
# 4. Session 후 .eval

# 변수 실행 방법들 3가지
# Session() 지정 후
# sess.run 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

aaa = sess.run(hypothesis)
print("aaa :", aaa)
sess.close()

# 세션을 InteractiveSession로 지정 한 다음에
# 베리어블.eval()을 사용하는 방법
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

bbb = hypothesis.eval()
print("bbb :",bbb)
sess.close()

# 세션을 사용하고
# 베리어블.eval()에 세션을 명시하는 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

ccc = hypothesis.eval(session=sess)
print("ccc :",ccc)
sess.close()
