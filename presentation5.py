import tensorflow as tf


# Model Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Input
x = tf.placeholder(tf.float32)

# Output
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# Calculating Loss
squared_delta = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

