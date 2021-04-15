import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session()
print(sess.run(hello))