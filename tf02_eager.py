# 1.xx 버전 처럼 사용하기 위해 즉시 실행 모드를 끄기 => disable_eager_execution
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

"""
print(tf.executing_eagerly()) #False

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) #False
import tensorflow as tf
print(tf.__version__)
"""

hello = tf.constant("hello World")
print(hello)

sess = tf.compat.v1.Session()
print(sess.run(hello))
