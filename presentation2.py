import tensorflow as tf


a = tf.constant(8.0)
b = tf.constant(7.0)

c = a * b

sess = tf.Session()

File_Writer = tf.summary.FileWriter('/Users/atiwari/Repos/OWN/TensorFlowPresentation/graph', sess.graph)

print(sess.run(c))

sess.close()