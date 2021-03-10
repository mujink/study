import tensorflow as tf
print(tf.__version__)

hello = tf.constant("hello World")
print(hello)

sess = tf.Session()
print(sess.run(hello))