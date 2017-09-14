import tensorflow as tf
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#building a computational graph
node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(4.0)

print(node1)
print(node2)

#running a computational graph
sess = tf.Session()
print(sess.run([node1, node2]))
